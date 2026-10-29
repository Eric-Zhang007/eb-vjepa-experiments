"""Wait for the 20k grid and verified archives, then run serial matched trials."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from grid_peak_search import decision
from grid_cache_storage import retire_grid_payloads, validate_archived_source
from run_encoder_candidate_queue import completed_results
from gpu_subset import parse_gpus, gpu_string


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def alive(pid, expected=None):
    path = Path(f"/proc/{pid}/stat")
    try:
        return (path.read_text().split(") ", 1)[1][0] != "Z"
                and (expected is None or expected.encode() in path.with_name("cmdline").read_bytes()))
    except FileNotFoundError:
        return False


def idle(gpus):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in rows.splitlines()}
    return all(used.get(gpu, 999999) <= 1000 for gpu in gpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-cache-step", type=int)
    parser.add_argument("--reuse-prior-10k", action="store_true",
                        help="Reuse verified prior 10k results, retaining their source identity")
    parser.add_argument("--max-encoder-step", type=int, default=20000, choices=range(2000, 20001, 1000))
    parser.add_argument("--allow-saved-early-stop", action="store_true")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if len(args.gpus) * args.batch_size != 256:
        raise ValueError("Matched global batch must remain 256")
    root = Path(__file__).resolve().parents[1]
    pretrain = root / "experiments/ctp_grid20k_20261008"
    out = root / "experiments/action_only/ctp_grid20k_search"
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (out / "search.log").open("a") as log:
            process = subprocess.Popen([sys.executable, __file__, *[x for x in sys.argv[1:] if x != "--detach"]],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"search_pid": process.pid}))
        return
    with (out / "search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "status.json"
        if path.exists() and not args.resume:
            raise RuntimeError("Search already initialized; explicit audited resume required")
        state = {"pid": os.getpid(), "started_unix": time.time(),
                 "objective": "0.5 * clean200 success rate + 0.5 * Plus280 success rate; fixed20k action head",
                 "grid": list(range(1000, args.max_encoder_step + 1, 1000)), "pretrain": str(pretrain),
                 "physical_gpus": args.gpus, "batch_size_per_gpu": args.batch_size}
        def status(**updates):
            state.update(updates, updated_unix=time.time())
            atomic(path, state)
        child = None
        def cancelled(*_):
            raise InterruptedError("Search terminated by signal")
        signal.signal(signal.SIGTERM, cancelled)
        signal.signal(signal.SIGINT, cancelled)
        try:
            launch = read(pretrain / "launch.json")
            while alive(launch["pid"], str(pretrain)):
                status(phase="waiting_for_pretrain", pretrain_pid=launch["pid"])
                time.sleep(30)
            summary = read(pretrain / "summary.json")
            if (not summary["sanity_passed_all_ranks"] or summary["step"] < args.max_encoder_step
                    or (not args.allow_saved_early_stop and summary["step"] != 20000)):
                raise RuntimeError("Pretraining did not safely save the requested grid")
            for step in range(1000, args.max_encoder_step + 1, 1000):
                receipt = pretrain / f"archive_receipt_{step:06d}.json"
                export = pretrain / f"embodied_vjepa_encoder_{step:06d}.pt"
                if not export.is_file():
                    raise RuntimeError(f"Missing EMA export: {step}")
                if receipt.exists() and read(receipt).get("phase") == "complete":
                    validate_archived_source(pretrain / f"checkpoint_audit_{step:06d}.json",
                                             export, step, pretrain / f"step_{step:06d}.pt")
                elif not (pretrain / f"step_{step:06d}.pt").is_file():
                    raise RuntimeError(f"Neither live nor archived source exists: {step}")
            scores, results = {}, []
            if args.reuse_prior_10k:
                prior = root / "experiments/action_only/ctp_full_010000_handoff"
                rows = completed_results(prior, 10000)
                for row in rows:
                    row["reused_from"] = str(prior)
                results.extend(rows)
                scores[10] = sum(r["success_rate"] for r in rows if r["selection"] == "fixed20k") / 2
                atomic(out / "reused_results.json", rows)
            for directory in sorted(out.glob("encoder_*")):
                if not directory.is_dir():
                    continue
                step = int(directory.name.split("_")[-1])
                if step == args.resume_cache_step:
                    if read(directory / 'continuation_status.json')['phase'] != 'preparing_cache':
                        raise RuntimeError('Resize only allowed during cache preparation')
                    continue
                if args.reuse_prior_10k and step == 10000:
                    retire_grid_payloads(directory, step, cancelled_replacement=True)
                    continue
                rows = completed_results(directory, step)
                for row in rows:
                    row["encoder_source"] = "ctp_grid20k_20261008"
                scores[step // 1000] = sum(r["success_rate"] for r in rows if r["selection"] == "fixed20k") / 2
                results.extend(rows)
                retire_grid_payloads(directory, step)
            atomic(out / "results.json", results)
            atomic(out / "scores.json", scores)
            while True:
                next_action = decision(scores, args.max_encoder_step // 1000)
                atomic(out / "decision.json", next_action)
                if next_action["next"] is None:
                    status(phase="complete", result=next_action, finished_unix=time.time())
                    break
                step = next_action["next"] * 1000
                run = out / f"encoder_{step:06d}"
                if run.exists() and step != args.resume_cache_step:
                    raise RuntimeError("Incomplete candidate exists; audit instead of duplicating")
                while not idle(args.gpus):
                    status(phase="waiting_for_idle_gpus", encoder_step=step)
                    time.sleep(30)
                command = [str(root / "envs/action_only_test/bin/python"), str(root / "scripts/continue_final_action.py"),
                           "--encoder-step", str(step), "--output", str(run), "--prepare", "--select-low-loss",
                           "--pretrain-output", str(pretrain), "--hybrid-cache",
                           "--run-name", f"libero_action_only_grid20261008_{step:06d}",
                           "--gpus", gpu_string(args.gpus), "--batch-size", str(args.batch_size)]
                receipt = pretrain / f"archive_receipt_{step:06d}.json"
                if receipt.exists() and read(receipt).get("phase") == "complete":
                    command.append("--archived-source")
                if step == args.resume_cache_step:
                    command.append("--resume-cache")
                status(phase="launching_candidate", encoder_step=step, decision=next_action, command=command)
                with (out / f"candidate_{step:06d}.log").open("a") as log:
                    env = {**os.environ, "TMPDIR": str(root / "cache/tmp"),
                           "PYTHONPATH": f"{root}/repos/RoboActualizer/src:{root}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages",
                           "CUDA_VISIBLE_DEVICES": gpu_string(args.gpus)}
                    child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=log, start_new_session=True)
                    while child.poll() is None:
                        candidate = run / "continuation_status.json"
                        status(phase="running_candidate", child_pid=child.pid,
                               candidate_phase=read(candidate)["phase"] if candidate.exists() else "starting")
                        time.sleep(30)
                    if child.returncode:
                        raise RuntimeError(f"Candidate {step} exited with {child.returncode}")
                child = None
                rows = completed_results(run, step)
                for row in rows:
                    row["encoder_source"] = "ctp_grid20k_20261008"
                scores[step // 1000] = sum(r["success_rate"] for r in rows if r["selection"] == "fixed20k") / 2
                results.extend(rows)
                atomic(out / "results.json", results)
                atomic(out / "scores.json", scores)
                lines = ["# Grid Search Results", "", state["objective"], "",
                         "Exploratory benchmark subsets used for selection, not held-out confirmation.", "",
                         "| Encoder step | Action step | Benchmark | Successes | Episodes | Rate |",
                         "| --- | --- | --- | --- | --- | --- |"]
                for row in results:
                    lines.append(f"| {row['encoder_step']} | {row['action_step']} | {row['benchmark']} | {row['successes']} | {row['episodes']} | {row['success_rate']:.4f} |")
                (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
                retire_grid_payloads(run, step)
        except BaseException as exc:
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=60)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            status(phase="cancelled" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()

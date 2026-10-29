"""Serial matched encoder trials; adopt existing 40k without restarting it."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

STEPS = (40000, 10000, 20000, 0, 30000, 50000, 60000)


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def completed_results(run, step):
    if read(run / "continuation_status.json")["phase"] != "complete":
        raise RuntimeError("Cannot aggregate or retire incomplete run")
    training = run / f"train_action_only_{step:06d}"
    selected = read(training / "checkpoint_selection.json")["selected"]["step"]
    results = []
    for action_step in dict.fromkeys([20000, selected]):
        suffix = "20k" if action_step == 20000 else f"low_loss_{action_step:06d}"
        checkpoint = training / f"checkpoints/weights/step_{action_step:06d}.pt"
        with checkpoint.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        for benchmark, expected in [("clean", 200), ("plus", 280)]:
            report = read(run / f"eval_{benchmark}_{suffix}/summary.json")
            records = report["records"]
            keys = {(r["suite"], r["task_id"], r["trial"]) for r in records}
            successes = sum(bool(r["success"]) for r in records)
            if (len(records) != expected or len(keys) != expected or report["episodes"] != expected
                    or report["checkpoint_sha256"] != sha or report["successes"] != successes):
                raise RuntimeError("Incomplete or mismatched evaluation summary")
            results.append({"encoder_step": step, "encoder_source": "official_original" if step == 0 else "full_droid_ctp", "action_step": action_step,
                            "selection": "fixed20k" if action_step == 20000 else "lowest_train_loss",
                            "benchmark": benchmark, "successes": successes, "episodes": expected,
                            "success_rate": successes / expected, "checkpoint_sha256": sha})
    return results


def retire_cache(base, step, authorized):
    if not authorized:
        raise RuntimeError("Cache retirement requires explicit authorization")
    if step not in STEPS:
        raise ValueError("Unapproved encoder step")
    base = base.resolve(strict=True)
    run = base / f"ctp_full_{step:06d}_handoff"
    completed_results(run, step)
    if read(run / "formal_train_status.json")["phase"] != "complete":
        raise RuntimeError("Training not complete")
    cache = run / f"cache_retry1/cache_final_{step:06d}_exclude_goal82"
    if cache.resolve(strict=True) != cache:
        raise RuntimeError("Refusing symlinked cache path")
    stats = read(cache / "latent_norm_stats.json")
    if stats["num_files"] != 1692:
        raise RuntimeError("Unexpected cache size")
    files = list(cache.glob("*/episode_*.pt"))
    if any(f.is_symlink() or not f.resolve().is_relative_to(cache)
           or not re.fullmatch(r"episode_\d+\.pt", f.name) for f in files):
        raise RuntimeError("Unsafe cache payload path")
    marker = run / "cache_payload_retirement.json"
    audit = read(marker) if marker.exists() else {
        "authorized": True, "target": str(cache), "original_payload_count": len(files),
        "original_bytes": sum(f.stat().st_size for f in files),
        "preserved": "All JSON statistics/identity, model weights, logs, evaluation results"}
    audit.update(phase="removing_payloads", started_unix=time.time())
    atomic(marker, audit)
    for path in files:
        path.unlink()
    audit.update(phase="complete", finished_unix=time.time())
    atomic(marker, audit)


def live_owner(status):
    pid = status.get("pid")
    path = Path(f"/proc/{pid}/cmdline")
    return isinstance(pid, int) and path.exists() and b"continue_final_action.py" in path.read_bytes()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "experiments/action_only"
    out = base / "encoder_candidate_queue"
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (out / "queue.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__, *[x for x in sys.argv[1:] if x != "--detach"]],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"queue_pid": child.pid}))
        return
    with (out / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = out / "status.json"
        if status_path.exists() and not args.resume:
            raise RuntimeError("Existing queue; use audited --resume, not duplicate launch")
        state = {"pid": os.getpid(), "steps": STEPS, "started_unix": time.time()}
        results = []
        def save(**updates):
            state.update(updates, updated_unix=time.time())
            atomic(status_path, state)
        def report():
            atomic(out / "results.json", results)
            lines = ["# Matched Encoder Trials", "", "Training-distribution candidate selection; exploratory benchmark subsets.",
                     "Fixed20k is the primary comparison; low-training-loss selection is secondary.", "",
                     "| Encoder | Action step | Selection | Benchmark | Success | Rate |",
                     "| --- | --- | --- | --- | --- | --- |"]
            for row in results:
                label = "official_original" if row["encoder_step"] == 0 else row["encoder_step"]
                lines.append(f"| {label} | {row['action_step']} | {row['selection']} | {row['benchmark']} | {row['successes']}/{row['episodes']} | {row['success_rate']:.1%} |")
            (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
        try:
            for index, step in enumerate(STEPS):
                run = base / f"ctp_full_{step:06d}_handoff"
                path = run / "continuation_status.json"
                save(phase="checking_candidate", encoder_step=step, run=str(run))
                if not path.exists():
                    if step == 40000:
                        raise RuntimeError("Expected existing 40k experiment")
                    if run.exists():
                        raise RuntimeError("Candidate directory exists without status; audit required")
                    # No cache deletion unless the user explicitly approves the policy.
                    while shutil.disk_usage(root).free < 260_000_000_000:
                        consent = out / "cache_recycling_authorization.json"
                        allowed = consent.exists() and read(consent).get("authorized") is True
                        if allowed:
                            retire_cache(base, STEPS[index - 1], True)
                            if shutil.disk_usage(root).free < 260_000_000_000:
                                raise RuntimeError("Still insufficient disk after approved cache retirement")
                            break
                        save(phase="waiting_for_cache_recycling_authorization", encoder_step=step)
                        time.sleep(30)
                    sources = ([root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"] if step == 0 else
                               [root / "experiments/ctp_full_72h" / name for name in
                                [f"step_{step:06d}.pt", f"embodied_vjepa_encoder_{step:06d}.pt"]])
                    for source in sources:
                        if not source.is_file():
                            raise RuntimeError(f"Missing candidate checkpoint: {source}")
                    command = [sys.executable, str(root / "scripts/continue_final_action.py"),
                               "--encoder-step", str(step), "--output", str(run), "--prepare", "--select-low-loss"]
                    save(phase="launching", command=command)
                    with (out / f"candidate_{step:06d}.log").open("a") as log:
                        child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                    save(phase="running", child_pid=child.pid)
                    for _ in range(30):
                        if path.exists():
                            break
                        if child.poll() is not None:
                            raise RuntimeError("Candidate exited before creating status")
                        time.sleep(1)
                    if not path.exists():
                        raise RuntimeError("Candidate status startup timeout")
                while True:
                    current = read(path)
                    save(phase="running", encoder_step=step, child_phase=current["phase"], child_pid=current["pid"])
                    if current["phase"] == "complete":
                        break
                    if current["phase"] == "failed" or not live_owner(current):
                        raise RuntimeError(f"Candidate {step} failed or lost owner: {current.get('error')}")
                    time.sleep(30)
                results.extend(completed_results(run, step))
                report()
            save(phase="complete", finished_unix=time.time())
        except Exception as exc:
            save(phase="failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()

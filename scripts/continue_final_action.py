"""Single-owner cache -> matched action training -> clean/Plus evaluation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from gpu_subset import parse_gpus, gpu_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder-step", type=int, default=103233)
    parser.add_argument("--pretrain-output", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--archived-source", action="store_true")
    parser.add_argument("--hybrid-cache", action="store_true")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume-cache", action="store_true")
    parser.add_argument("--wait-for-run", type=Path)
    parser.add_argument("--select-low-loss", action="store_true")
    parser.add_argument("--concurrent-cache", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) * args.batch_size != 256:
        raise ValueError("Matched action global batch must remain 256")
    root = Path(__file__).resolve().parents[1]
    out = args.output or root / "experiments/action_only/ctp_full_handoff"
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (out / "continuation.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__, *[x for x in sys.argv[1:] if x != "--detach"]], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"pid": child.pid}))
        return
    lock = (out / "continuation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = out / "continuation_status.json"
    if path.exists():
        previous = json.loads(path.read_text())
        proc = Path(f"/proc/{previous['pid']}/stat")
        if (not args.resume_cache or previous['phase'] != 'preparing_cache'
                or (proc.exists() and proc.read_text().split(') ', 1)[1][0] != 'Z')
                or (out / 'formal_train_status.json').exists()):
            lock.close()
            raise RuntimeError("Existing continuation; refusing duplicate or unsafe restart")
        (out / f"continuation_before_resize_{time.time_ns()}.json").write_text(path.read_text())
    state = {"pid": os.getpid(), "started_unix": time.time()}
    def save(**updates):
        state.update(updates)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)
    children = []
    try:
        cache_out = out / "cache_retry1"
        train_out = out / f"train_action_only_{args.encoder_step:06d}"
        pretrain = args.pretrain_output or root / "experiments/ctp_full_72h"
        encoder = pretrain / f"embodied_vjepa_encoder_{args.encoder_step:06d}.pt"
        if args.encoder_step == 0:
            encoder = root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
        if args.wait_for_run and not args.concurrent_cache:
            save(phase="waiting_for_previous_evaluation", predecessor=str(args.wait_for_run))
            while True:
                predecessor = json.loads((args.wait_for_run / "continuation_status.json").read_text())
                if predecessor["phase"] == "complete":
                    break
                if predecessor["phase"] == "failed":
                    raise RuntimeError("Previous evaluation failed; audit before proceeding")
                if not Path(f"/proc/{predecessor['pid']}").exists():
                    raise RuntimeError("Previous evaluation owner exited without completion")
                time.sleep(30)
        if args.prepare:
            command = [sys.executable, str(root / "scripts/prepare_final_cache.py"),
                       "--encoder", str(encoder), "--step", str(args.encoder_step),
                       "--source-checkpoint", str(pretrain / f"step_{args.encoder_step:06d}.pt"),
                       "--output", str(cache_out), "--workers-per-gpu", "4", "--stats-workers", "32",
                       "--gpus", gpu_string(args.gpus)]
            if args.concurrent_cache:
                command.append("--share-evaluation-gpus")
            if args.resume_cache:
                command.append("--resume")
            if args.archived_source:
                command += ["--source-audit", str(pretrain / f"checkpoint_audit_{args.encoder_step:06d}.json")]
            if args.hybrid_cache:
                command.append("--hybrid-cache")
            if args.encoder_step == 0:
                command.append("--official-original")
            save(phase="preparing_cache", command=command)
            subprocess.run(command, check=True, cwd=root)
        save(phase="waiting_for_cache", cache=str(cache_out), train=str(train_out))
        while True:
            cache_state = json.loads((cache_out / "full_cache_status.json").read_text())
            if cache_state["phase"] == "complete":
                break
            if cache_state["phase"] == "failed":
                raise RuntimeError(f"Cache preparation failed: {cache_state}")
            if not Path(f"/proc/{cache_state['pid']}").exists():
                raise RuntimeError("Cache owner exited without completion")
            time.sleep(30)
        if args.wait_for_run and args.concurrent_cache:
            save(phase="waiting_for_evaluation_before_training")
            while True:
                predecessor = json.loads((args.wait_for_run / "continuation_status.json").read_text())
                if predecessor["phase"] == "complete":
                    break
                if predecessor["phase"] == "failed" or not Path(f"/proc/{predecessor['pid']}").exists():
                    raise RuntimeError("Previous evaluation failed or exited; audit before training")
                time.sleep(30)
        command = [sys.executable, str(root / "scripts/launch_action_final.py"),
                   "--encoder", str(encoder), "--step", str(args.encoder_step),
                   "--cache-output", str(cache_out), "--train-output", str(train_out),
                   "--gpus", gpu_string(args.gpus), "--batch-size", str(args.batch_size)]
        if args.run_name:
            command += ["--run-name", args.run_name]
        save(phase="training", train_command=command)
        subprocess.run(command, check=True, cwd=root)
        # Match the previously evaluated policy checkpoint, not a selected best score.
        from select_action_checkpoint import select_checkpoint
        selected = select_checkpoint(train_out) if args.select_low_loss else 20000
        steps = list(dict.fromkeys([20000, selected]))
        checkpoint = train_out / "checkpoints/weights/step_020000.pt"
        if not checkpoint.is_file():
            raise RuntimeError("Matched 20k action checkpoint missing")
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for eval_step, kind, count in [(step, kind, count) for step in steps
                                       for kind, count in (("clean", 200), ("plus", 280))]:
            checkpoint = train_out / f"checkpoints/weights/step_{eval_step:06d}.pt"
            sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            suffix = "20k" if eval_step == 20000 else f"low_loss_{eval_step:06d}"
            target = out / f"eval_{kind}_{suffix}"
            target.mkdir(exist_ok=False)
            if kind == "plus":
                source = root / "experiments/action_only/libero_20k_plus_280_shared_run"
                for name in ("manifest.json", "preflight.json"):
                    shutil.copy2(source / name, target / name)
                (target / "text_cache").symlink_to((source / "text_cache").resolve(), target_is_directory=True)
                manifest = json.loads((target / "manifest.json").read_text())
                assert manifest["episodes"] == count
            save(phase=f"evaluating_{kind}", evaluation_checkpoint=str(checkpoint),
                 checkpoint_sha256=sha, workers=[])
            children = []
            for gpu in range(8):
                command = ["bash", str(root / "scripts/run_10k_libero_probe.sh"),
                           "--shard", str(gpu), "--step", str(eval_step), "--run", str(train_out),
                           "--output", str(target)]
                # Preserve eight protocol shards, mapped only onto owned devices.
                env = {**os.environ, "EVAL_GPU": str(args.gpus[gpu % len(args.gpus)])}
                if kind == "plus":
                    command += ["--benchmark", "plus", "--shards", "8", "--text-cfg-scale", "1.0"]
                    env.update(EVAL_SCRIPT="eval_10k_robust.py",
                               LIBERO_SOURCE=str(root / "repos/LIBERO-plus-official"),
                               LIBERO_CONFIG_PATH=str(root / "experiments/action_only/plus_config"))
                else:
                    env.update(EVAL_SCRIPT="eval_10k_libero_probe.py",
                               LIBERO_SOURCE="/data/users/jiachangzhang/DNFT4RP/upstream/libero",
                               LIBERO_CONFIG_PATH="/data/users/jiachangzhang/DNFT4RP/.libero-config")
                with (target / f"gpu_{gpu}.log").open("a") as log:
                    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=log)
                children.append(child)
                state["workers"].append({"gpu": args.gpus[gpu % len(args.gpus)], "shard": gpu,
                                         "pid": child.pid, "command": command})
                save()
            while any(p.poll() is None for p in children):
                if any(p.poll() not in (None, 0) for p in children):
                    raise RuntimeError(f"{kind} evaluation worker failed")
                time.sleep(15)
            if any(p.returncode for p in children):
                raise RuntimeError(f"{kind} evaluation failed")
            records = []
            for gpu in range(8):
                report = json.loads((target / f"shard_{gpu}/report.json").read_text())
                assert report["phase"] == "complete" and report["checkpoint_sha256"] == sha
                records.extend(report["tasks" if kind == "clean" else "episodes"])
            assert len(records) == count
            summary = {"episodes": count, "successes": sum(bool(r["success"]) for r in records),
                       "checkpoint_sha256": sha, "records": records,
                       "protocol": "Matched exploratory subset; not full benchmark"}
            summary["success_rate"] = summary["successes"] / count
            (target / "summary.json").write_text(json.dumps(summary, indent=2))
        save(phase="complete", finished_unix=time.time())
    except Exception as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        save(phase="failed", error=repr(exc), finished_unix=time.time())
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()

"""Single-owner eight-GPU evaluation supervisor; no training mutations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--step", type=int, default=10000, choices=(10000, 20000))
    args = parser.parse_args()
    root = Path("/data/users/jiachangzhang/eb_vjepa")
    out = root / f"experiments/action_only/libero_{args.step // 1000}k_egl_200"
    out.mkdir(exist_ok=True)
    if args.detach:
        with (out / "supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__, "--step", str(args.step)], start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        print(json.dumps({"supervisor_pid": child.pid}))
        return
    lock = (out / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = out / "status.json"
    if status_path.exists():
        raise RuntimeError("Existing evaluation launch; refusing duplicate")
    preflight = root / "experiments/action_only/libero_10k_egl_200/preflight.json"
    assert json.loads(preflight.read_text())["status"] == "PASS"
    checkpoint = root / f"experiments/action_only/train_final/checkpoints/weights/step_{args.step:06d}.pt"
    assert checkpoint.is_file()
    status = {"phase": "launching", "pid": os.getpid(), "started_unix": time.time(),
              "checkpoint_step": args.step, "preflight": str(preflight),
              "planned_episodes": 200, "workers": []}
    def save():
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(status_path)
    children = []
    save()
    try:
        for gpu in range(8):
            command = ["bash", str(root / "scripts/run_10k_libero_probe.sh"),
                       "--shard", str(gpu), "--step", str(args.step)]
            with (out / f"gpu_{gpu}.log").open("a") as log:
                child = subprocess.Popen(command, env={**os.environ, "EVAL_GPU": str(gpu)},
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            children.append(child)
            status["workers"].append({"gpu": gpu, "pid": child.pid, "command": command})
            save()
            time.sleep(2)
        status["phase"] = "running"
        save()
        while True:
            codes = [p.poll() for p in children]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"Worker failure: {codes}")
            if all(code == 0 for code in codes):
                break
            time.sleep(10)
        tasks = []
        for gpu in range(8):
            report = json.loads((out / f"shard_{gpu}/report.json").read_text())
            assert report["phase"] == "complete" and len(report["tasks"]) == 25
            tasks.extend(report["tasks"])
        assert len({(x["suite"], x["task_id"], x["trial"]) for x in tasks}) == 200
        summary = {"tasks": tasks, "episodes": 200, "checkpoint_step": args.step,
                   "successes": sum(x["success"] for x in tasks)}
        summary["success_rate"] = summary["successes"] / 200
        summary["suites"] = {suite: {
            "successes": sum(x["success"] for x in tasks if x["suite"] == suite), "episodes": 50}
            for suite in sorted({x["suite"] for x in tasks})}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        status.update(phase="complete", finished_unix=time.time(), successes=summary["successes"])
        save()
    except Exception as exc:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        status.update(phase="failed", error=repr(exc), finished_unix=time.time())
        save()
        raise


if __name__ == "__main__":
    main()

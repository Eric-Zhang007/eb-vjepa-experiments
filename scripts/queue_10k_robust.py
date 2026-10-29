"""Sequential clean -> Plus -> Pro evaluation queue with separate source imports."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
OUT = ROOT / "experiments/action_only/robust_10k_queue"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--retry-failed-preflight", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.detach:
        with (OUT / "supervisor.log").open("a") as log:
            command = [sys.executable, __file__]
            if args.retry_failed_preflight:
                command.append("--retry-failed-preflight")
            p = subprocess.Popen(command, start_new_session=True,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        print(json.dumps({"supervisor_pid": p.pid}))
        return
    lock = (OUT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT / "status.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if not (args.retry_failed_preflight and previous["phase"] == "failed"
                and "preflight failed" in previous.get("error", "")
                and all("worker_pids" not in s for s in previous["stages"].values())):
            raise RuntimeError("Existing queue; refusing duplicate")
        path.rename(OUT / f"status_failed_{previous['pid']}.json")
    state = {"phase": "waiting_clean", "pid": os.getpid(), "started_unix": time.time(), "stages": {}}
    def save():
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)
    children = []
    save()
    def spawn(kind, gpu, prepare=False):
        source = ROOT / "repos" / ("LIBERO-plus-official" if kind == "plus" else "LIBERO-PRO-official")
        env = {**os.environ, "EVAL_GPU": str(gpu), "EVAL_SCRIPT": "eval_10k_robust.py",
               "LIBERO_SOURCE": str(source),
               "LIBERO_CONFIG_PATH": str(ROOT / f"experiments/action_only/{kind}_config")}
        command = ["bash", str(ROOT / "scripts/run_10k_libero_probe.sh"), "--benchmark", kind]
        command += ["--prepare"] if prepare else ["--shard", str(gpu)]
        with (OUT / f"{kind}_{'prepare' if prepare else gpu}.log").open("a") as log:
            p = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        return p
    try:
        clean_path = ROOT / "experiments/action_only/libero_10k_egl_200/status.json"
        while True:
            clean = json.loads(clean_path.read_text())
            if clean["phase"] == "failed":
                raise RuntimeError("Clean evaluation failed")
            if clean["phase"] == "complete":
                break
            time.sleep(20)
        for kind in ("plus", "pro"):
            state["phase"] = "preparing_" + kind
            p = spawn(kind, 7, prepare=True)
            children = [p]
            state["stages"][kind] = {"prepare_pid": p.pid}
            save()
            if p.wait() != 0:
                raise RuntimeError(kind + " preflight failed; no benchmark score")
        for kind in ("plus", "pro"):
            state["phase"] = "running_" + kind
            children = []
            for gpu in range(8):
                children.append(spawn(kind, gpu))
                time.sleep(1)
            state["stages"][kind]["worker_pids"] = [p.pid for p in children]
            save()
            while True:
                codes = [p.poll() for p in children]
                if any(c not in (None, 0) for c in codes):
                    raise RuntimeError(f"{kind} worker failure {codes}; see partial reports")
                if all(c == 0 for c in codes):
                    break
                time.sleep(10)
            folder = ROOT / f"experiments/action_only/libero_10k_{kind}"
            manifest = json.loads((folder / "manifest.json").read_text())
            episodes = []
            for gpu in range(8):
                report = json.loads((folder / f"shard_{gpu}/report.json").read_text())
                assert report["phase"] == "complete" and len(report["episodes"]) == report["planned_episodes"]
                episodes.extend(report["episodes"])
            assert len(episodes) == manifest["episodes"]
            assert len({(e["index"], e["trial"]) for e in episodes}) == len(episodes)
            summary = {"episodes": episodes, "total": len(episodes),
                       "successes": sum(e["success"] for e in episodes), "checkpoint_step": 10000}
            summary["success_rate"] = summary["successes"]/summary["total"]
            summary["categories"] = {cat: {
                "episodes": sum(e["category"] == cat for e in episodes),
                "successes": sum(e["success"] for e in episodes if e["category"] == cat)}
                for cat in sorted({e["category"] for e in episodes})}
            (folder / "summary.json").write_text(json.dumps(summary, indent=2))
            state["stages"][kind]["complete"] = True
            state["stages"][kind]["successes"] = summary["successes"]
            save()
        state.update(phase="complete", finished_unix=time.time())
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
        state.update(phase="failed", error=repr(exc), finished_unix=time.time())
        save()
        raise


if __name__ == "__main__":
    main()

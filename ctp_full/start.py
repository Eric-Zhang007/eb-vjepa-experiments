"""Detach the bounded acceptance pipeline; formal launch is gated inside launch.py."""
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
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "experiments/ctp_full_72h"
    if args.detach:
        with (out / "supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid}), flush=True)
        return
    lock = (out / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = out / "pipeline_status.json"
    if path.exists():
        raise RuntimeError("Existing pipeline status: inspect, do not duplicate")
    state = {"phase": "acceptance", "pid": os.getpid(), "started_unix": time.time()}
    def save(**extra):
        state.update(extra)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)
    save()
    env = {**os.environ, "TMPDIR": str(root / "cache/tmp")}
    command = [sys.executable, str(Path(__file__).with_name("launch.py")), "--launch"]
    try:
        with (out / "pipeline.log").open("a") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log)
            save(child_pid=process.pid, command=command)
            code = process.wait()
        if code:
            raise RuntimeError(f"Acceptance/launch failed with exit {code}; inspect pipeline.log")
        launch = json.loads((out / "launch.json").read_text())
        save(phase="training", formal_pid=launch["pid"], launched_unix=time.time())
    except Exception as exc:
        save(phase="failed", error=str(exc), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()

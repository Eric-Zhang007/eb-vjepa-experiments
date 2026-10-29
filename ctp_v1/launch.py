"""One bounded calibration, one 90+10 resume smoke, optionally launch training."""
import argparse
import datetime
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import yaml


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def ensure_idle():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    if result.stdout.strip():
        raise RuntimeError("GPU compute processes already exist; refusing a competing run")


def run_logged(command, log, env, timeout):
    with open(log, "w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only terminate the process group that this launcher created.
            import signal
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--launch", action="store_true", help="Launch formal run after all gates pass")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = Path(config["root"])
    output = root / config["output"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "pipeline.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = root / config["checkpoint"]
    if not source.is_file() or source.stat().st_size != config["checkpoint_bytes"]:
        raise RuntimeError("Checkpoint download incomplete. No calibration or training started.")
    if not (root / config["manifest"]).is_file():
        raise FileNotFoundError("Build the PyAV episode manifest with check_data.py first")
    if args.launch and any((output / name).exists() for name in ("latest.pt", "metrics.jsonl", "launch.json")):
        raise RuntimeError("Formal run already initialized; inspect it and resume train.py explicitly")
    ensure_idle()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "repos/vjepa2")
    for key, subdir in (
        ("TMPDIR", "cache/tmp"), ("CUDA_CACHE_PATH", "cache/cuda"),
        ("TORCH_HOME", "cache/torch"), ("TRITON_CACHE_DIR", "cache/triton"),
        ("XDG_CACHE_HOME", "cache/xdg"),
    ):
        directory = root / subdir
        directory.mkdir(parents=True, exist_ok=True)
        env[key] = str(directory)
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    (output / "topology.txt").write_text(topology)
    trainer = Path(__file__).with_name("train.py")
    command = [str(Path(sys.executable).with_name("torchrun")), "--standalone",
               "--nproc_per_node=8", str(trainer)]
    calibration = output / "calibration"
    calibration.mkdir(exist_ok=True)
    selection_path = calibration / "selection.json"
    calibrated_config = output / "calibrated.yaml"
    if not selection_path.exists():
        if (calibration / "started.json").exists():
            raise RuntimeError("Prior bounded calibration is incomplete; inspect logs before retrying")
        (calibration / "started.json").write_text(json.dumps({"started_at": stamp()}))
        results, blocked_modes = [], set()
        budget_start = time.monotonic()
        for microbatch in (4, 8, 16):
            for activation in ("on", "off"):
                if activation in blocked_modes:
                    continue
                remaining = 3600 - (time.monotonic() - budget_start)
                if remaining < 120:
                    raise TimeoutError("One-hour bounded calibration budget exhausted")
                trial = calibration / f"mb{microbatch}_ac_{activation}"
                trial.mkdir()
                trial_command = command + [
                    "--config", str(args.config), "--stage", "calibrate",
                    "--output", str(trial), "--microbatch", str(microbatch),
                    "--accumulation", str(16 // microbatch),
                    "--activation-checkpointing", activation, "--stop-step", "40",
                ]
                print("Calibration:", shlex.join(trial_command), flush=True)
                rc = run_logged(trial_command, trial / "console.log", env, min(1200, remaining))
                if rc:
                    text = (trial / "console.log").read_text(errors="replace")
                    if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
                        blocked_modes.add(activation)
                        (trial / "oom.json").write_text(json.dumps({"oom": True}))
                        continue
                    raise RuntimeError(f"Calibration failed (not OOM): {trial / 'console.log'}")
                summary = json.loads((trial / "summary.json").read_text())
                summary["microbatch"] = microbatch
                if not summary["sanity_passed_all_ranks"]:
                    raise RuntimeError(f"Calibration gradient/EMA sanity failed: {trial}")
                if summary["max_allocated_bytes"] > 0.90 * summary["device_total_bytes"]:
                    blocked_modes.add(activation)
                    continue
                results.append(summary)
        if not results:
            raise RuntimeError("No stable calibration candidate with sufficient headroom")
        best = max(results, key=lambda value: value["samples_per_second"])
        selection = {
            key: best[key] for key in (
                "microbatch", "accumulation", "activation_checkpointing",
                "samples_per_second", "mean_step_seconds", "expected_steps_72h",
            )
        }
        selection["candidates"] = results
        selection["selected_at"] = stamp()
        selection_path.write_text(json.dumps(selection, indent=2))
    selection = json.loads(selection_path.read_text())
    for key in ("microbatch", "accumulation", "activation_checkpointing"):
        config[key] = selection[key]
    calibrated_config.write_text(yaml.safe_dump(config, sort_keys=False))

    smoke = output / "smoke"
    smoke.mkdir(exist_ok=True)
    smoke_command = command + ["--config", str(calibrated_config), "--stage", "smoke",
                               "--output", str(smoke)]
    if not (smoke / "pass.json").exists():
        if (smoke / "latest.pt").exists():
            import torch
            saved = torch.load(smoke / "latest.pt", weights_only=False, map_location="cpu", mmap=True)
            saved_step = saved["step"]
            del saved
            if saved_step != 90:
                raise RuntimeError("Unexpected partial smoke: inspect rather than rerunning 100 steps")
        else:
            rc = run_logged(smoke_command + ["--stop-step", "90"],
                            smoke / "first90.log", env, 7200)
            if rc:
                raise RuntimeError(f"Smoke failed: {smoke / 'first90.log'}")
        rc = run_logged(smoke_command + ["--stop-step", "100", "--resume", str(smoke / "latest.pt")],
                        smoke / "resume10.log", env, 1800)
        if rc:
            raise RuntimeError(f"Resume smoke failed: {smoke / 'resume10.log'}")
    evidence = json.loads((smoke / "pass.json").read_text())
    if not evidence["passed"]:
        raise RuntimeError("100-step smoke did not pass")
    if not args.launch:
        print("Calibration and 100-step resume smoke passed; --launch enables formal start.", flush=True)
        return
    ensure_idle()
    formal = command + ["--config", str(calibrated_config), "--stage", "train"]
    log = output / "train.log"
    with log.open("a") as handle:
        process = subprocess.Popen(formal, stdout=handle, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    started = stamp()
    launch = {
        "pid": process.pid, "started_at": started, "command": shlex.join(formal),
        "environment": {k: env[k] for k in (
            "PYTHONPATH", "TMPDIR", "CUDA_CACHE_PATH", "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME")},
        "log": str(log), "stop": "30000 optimizer steps or 72 hours, whichever occurs first",
        "selection": selection,
    }
    (output / "launch.json").write_text(json.dumps(launch, indent=2))
    # Verify the background process reaches real training before reporting launch.
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError(f"Formal process exited early; see {log}")
        if (output / "metrics.jsonl").exists():
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"Formal PID {process.pid} is alive but first step not yet observed; see {log}")
    wandb = json.loads((output / "wandb.json").read_text())
    previous = output / "RUN.md"
    if previous.exists():
        (output / "PRELAUNCH.md").write_text(previous.read_text())
    previous.write_text(
        "# CTP-v1 Formal Run\n\n"
        f"Started (UTC): {started}\n\nPID / session leader: {process.pid}\n\n"
        f"Log: `{log}`\n\nW&B: `{wandb}`\n\n"
        f"Microbatch/GPU: {config['microbatch']}; accumulation: {config['accumulation']}; "
        f"effective global batch: {config['microbatch'] * config['accumulation'] * 8}.\n\n"
        f"Activation checkpointing: {config['activation_checkpointing']}.\n\n"
        f"Measured clips/sec: {selection['samples_per_second']:.3f}; "
        f"seconds/optimizer-step: {selection['mean_step_seconds']:.3f}; "
        f"projected 72h steps: {selection['expected_steps_72h']}.\n\n"
        "Stop at 30,000 optimizer steps or 72 hours. Save latest, every 5k, and final.\n\n"
        "Both encoders initialized from ema_encoder. Predictor trunk reused; "
        "two teacher-space output heads explicitly reinitialized for native 1024-dim EMA targets. "
        "See load_report_rank0.json for exact reuse percentages.\n\n"
        "Exact command (environment saved in launch.json):\n\n"
        f"```bash\n{shlex.join(formal)}\n```\n"
    )
    print(json.dumps(launch, indent=2), flush=True)


if __name__ == "__main__":
    main()

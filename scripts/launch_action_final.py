"""Explicitly authorized single-owner production action-only training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from gpu_subset import parse_gpus, gpu_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size * len(args.gpus) != 256:
        raise ValueError("Matched action recipe requires global batch 256")
    root = Path(__file__).resolve().parents[1]
    out = args.train_output.parent
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (out / "formal_supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__, *[x for x in sys.argv[1:] if x != "--detach"]], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid}))
        return
    lock = (out / "formal_train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = out / "formal_train_status.json"
    if state_path.exists() or args.train_output.exists() or (out / "ready_plan.json").exists():
        lock.close()
        raise RuntimeError("Existing formal run or watcher plan; refusing duplicate launch")
    state = {"phase": "validating", "supervisor_pid": os.getpid(), "started_unix": time.time()}
    def status(**extra):
        state.update(extra)
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)
    status()
    try:
        prepared = json.loads((args.cache_output / "full_cache_status.json").read_text())
        if prepared["phase"] != "complete":
            raise RuntimeError("Final cache is not complete")
        with args.encoder.open("rb") as stream:
            encoder_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if encoder_sha != prepared["encoder_sha256"]:
            raise RuntimeError("Requested encoder differs from prepared cache")
        manifest = root / "experiments/libero_episode_selection.json"
        selection = hashlib.sha256(manifest.read_bytes()).hexdigest()
        cache = Path(prepared["cache"])
        stats = json.loads((cache / "latent_norm_stats.json").read_text())
        if stats["episode_selection_sha256"] != selection or stats["num_files"] != 1692:
            raise RuntimeError("Production stats/selection mismatch")
        if stats["encoder_checkpoint_sha256"] != encoder_sha:
            raise RuntimeError("Normalization stats belong to another encoder")
        frames = 0
        for path in cache.glob("*/latents_meta.json"):
            meta = json.loads(path.read_text())
            if meta["episode_selection_sha256"] != selection:
                raise RuntimeError("Cache selection mismatch")
            if meta["encoder_checkpoint_sha256"] != prepared["encoder_sha256"]:
                raise RuntimeError("Encoder mismatch")
            if "goal" in path.parent.name and "82" in meta["episodes"]:
                raise RuntimeError("Excluded episode present")
            frames += sum(meta["episodes"].values())
        if frames != 273336:
            raise RuntimeError("Unexpected cache frame count")
        memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                          "--format=csv,noheader,nounits"], text=True)
        used = {int(line.split(",")[0]): int(line.split(",")[1]) for line in memory.splitlines()}
        if any(used.get(gpu, 999999) > 1000 for gpu in args.gpus):
            raise RuntimeError("Requested GPUs must be idle before launch")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(args.gpus)}",
                   str(root / "repos/RoboActualizer/scripts/train.py"),
                   "task=libero_action_only_final", f"batch_size={args.batch_size}", "num_workers=4",
                   "num_epochs=20", "gradient_accumulation_steps=1",
                   f"encoder_tag={'official_original' if args.step == 0 else f'ctp_final_{args.step:06d}'}",
                   f"model.backbone.ckpt_path={args.encoder}",
                   f"data.train.rgb_latent_cache_dir={cache}",
                   f"model.latent_norm_stats_path={cache}/latent_norm_stats.json",
                   f"wandb.name={'libero_action_only_official_original' if args.step == 0 else f'libero_action_only_ctp_final_{args.step:06d}'}",
                   f"output_dir={args.train_output}",
                   f"hydra.run.dir={args.train_output}/hydra"]
        if args.run_name:
            command = [f"wandb.name={args.run_name}" if item.startswith("wandb.name=") else item
                       for item in command]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_string(args.gpus), "OMP_NUM_THREADS": "4",
               "TMPDIR": str(root / "cache/tmp"), "HF_HOME": str(root / "cache/huggingface"),
               "HF_DATASETS_CACHE": str(root / "cache/huggingface/datasets"),
               "WANDB_DIR": str(root / "wandb"), "WANDB_MODE": "online",
               "PYTHONPATH": f"{root}/repos/RoboActualizer/src:{root}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"}
        with (out / "train_final.log").open("a") as log:
            child = subprocess.Popen(command, cwd=root / "repos/RoboActualizer", env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            status(phase="running", child_pid=child.pid, command=command, world_size=len(args.gpus),
                   physical_gpus=args.gpus, batch_size_per_gpu=args.batch_size,
                   global_batch=256, selection_sha256=selection, encoder_sha256=prepared["encoder_sha256"],
                   log=str(out / "train_final.log"))
            code = child.wait()
        status(phase="complete" if code == 0 else "failed", returncode=code, finished_unix=time.time())
        if code:
            raise RuntimeError(f"Action training exited with status {code}")
    except Exception as exc:
        status(phase="failed", error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()

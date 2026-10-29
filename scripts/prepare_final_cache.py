"""Single-owner full LIBERO final-EMA cache preparation, with persistent status."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from gpu_subset import parse_gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--hybrid-cache", action="store_true")
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5,6,7"))
    parser.add_argument("--official-original", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-retired", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=1, choices=(1, 2, 4))
    parser.add_argument("--normalize-only", action="store_true")
    parser.add_argument("--stats-workers", type=int, default=32)
    parser.add_argument("--share-evaluation-gpus", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["TMPDIR"] = str(root / "cache/tmp")
    out = args.output or root / "experiments/action_only"
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (out / "full_cache.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__, *[x for x in sys.argv[1:] if x != "--detach"]], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"pid": child.pid}))
        return
    lock = (out / "full_cache.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = out / "full_cache_status.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        rebuild = (args.rebuild_retired and args.step == 10000
                   and out == root / "experiments/action_only/ctp_full_010000_handoff/cache_retry1"
                   and previous["phase"] == "complete"
                   and not list(Path(previous['cache']).glob('*/episode_*.pt')))
        if not args.resume or (previous["phase"] == "complete" and not rebuild):
            raise RuntimeError("Prior preparation status exists; audit before restarting")
        for pid in [previous["pid"], *[w["pid"] for w in previous.get("workers", [])]]:
            proc = Path(f"/proc/{pid}/stat")
            if proc.exists() and proc.read_text().split(") ", 1)[1][0] != "Z":
                raise RuntimeError(f"Previous cache process still alive: {pid}")
        (out / f"full_cache_status_before_{time.time_ns()}.json").write_text(state_path.read_text())
    state = {"pid": os.getpid(), "phase": "validating", "started_unix": time.time()}
    def status(**extra):
        state.update(extra)
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(state_path)
    status()
    children = []
    logs = []
    try:
        if args.normalize_only:
            if not args.resume or previous["phase"] != "normalizing":
                raise RuntimeError("Normalization restart requires audited normalizing state")
            cache = Path(previous["cache"])
            with args.encoder.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != previous["encoder_sha256"]:
                raise RuntimeError("Normalization restart encoder mismatch")
            status(phase="normalizing", cache=str(cache), encoder_sha256=digest,
                   stats_workers=args.stats_workers)
            subprocess.run([sys.executable, str(root / "repos/RoboActualizer/scripts/compute_latent_norm_stats.py"),
                            str(cache), "--workers", str(args.stats_workers)],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                "PYTHONPATH": f"{root}/repos/RoboActualizer/src:{root}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"},
                           check=True)
            status(phase="complete", finished_unix=time.time())
            return
        import torch
        final = args.source_checkpoint or root / "experiments/ctp_full_72h/final.pt"
        if args.official_original:
            official = root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
            if args.step != 0 or args.encoder.resolve() != official.resolve():
                raise RuntimeError("Original baseline must use official checkpoint and step0 sentinel")
            final = official
        encoder = args.encoder or final.parent / f"embodied_vjepa_encoder_{args.step or 0:06d}.pt"
        export = torch.load(encoder, map_location="cpu", weights_only=False, mmap=True)
        checkpoint = None
        if args.source_audit:
            from grid_cache_storage import validate_archived_source
            audit = validate_archived_source(args.source_audit, encoder, args.step, final)
            step = audit["step"]
        else:
            checkpoint = torch.load(final, map_location="cpu", weights_only=False, mmap=True)
            step = 0 if args.official_original else int(checkpoint["step"])
        if args.step is not None and step != args.step:
            raise RuntimeError("Final CTP step does not match requested step")
        if checkpoint is not None and not args.official_original and not checkpoint.get("sanity_passed"):
            raise RuntimeError("CTP final did not pass training sanity checks")
        b = export["ema_encoder"]
        a = checkpoint["ema_encoder"] if checkpoint is not None else None
        if not all(torch.isfinite(value).all() for value in b.values()):
            raise RuntimeError("Nonfinite EMA export")
        if a is not None and (set(a) != set(b) or not all(torch.equal(a[k], b[k]) for k in a)):
            raise RuntimeError("Final EMA export differs from final training state")
        digest = hashlib.file_digest(encoder.open("rb"), "sha256").hexdigest()
        if args.official_original and digest != "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6":
            raise RuntimeError("Official checkpoint content hash differs from verified source")
        (out / "final_encoder_validation.json").write_text(json.dumps(
            {"status": "PASS", "step": step, "encoder": str(encoder), "sha256": digest,
             "source_kind": "official_original" if args.official_original else "ctp",
             "tensors": len(b), "export_equals_source_ema": True, "finite": True,
             "source_validation": "prior_hash_bound_audit_and_archive_receipt" if args.source_audit else "live_tensor_comparison"}, indent=2))
        del checkpoint, export, a, b
        free = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                        "--format=csv,noheader,nounits"], text=True)
        memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in free.splitlines()}
        if args.share_evaluation_gpus:
            free_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                                  "--format=csv,noheader,nounits"], text=True)
            free_mib = {int(line.split(",")[0]): int(line.split(",")[1]) for line in free_rows.splitlines()}
            if any(free_mib.get(gpu, 0) < 32768 for gpu in args.gpus):
                raise RuntimeError("Shared cache requires at least 32 GiB free per GPU")
        elif any(memory.get(gpu, 999999) > 1000 for gpu in args.gpus):
            raise RuntimeError("Requested cache GPUs must be idle")
        import shutil
        cache = out / f"cache_final_{step:06d}_exclude_goal82"
        existing_bytes = sum(p.stat().st_size for p in cache.glob("*/episode_*.pt"))
        ram_reservation = 0
        if args.hybrid_cache:
            from grid_cache_storage import prepare_hybrid_storage
            ram_reservation = prepare_hybrid_storage(root, cache, step)
            existing_bytes = sum(p.stat().st_size for p in cache.glob("*/episode_*.pt")
                                 if p.resolve().is_relative_to(root.resolve()))
        if shutil.disk_usage(root).free < max(40_000_000_000, 260_000_000_000 - ram_reservation - existing_bytes):
            raise RuntimeError("Insufficient headroom for the 220 GB full cache")
        env = {**os.environ, "OMP_NUM_THREADS": "4", "TMPDIR": str(root / "cache/tmp"),
               "HF_HOME": str(root / "cache/huggingface"),
               "HF_DATASETS_CACHE": str(root / "cache/huggingface/datasets"),
               "PYTHONPATH": f"{root}/repos/RoboActualizer/src:{root}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"}
        commands = []
        if len(args.gpus) * args.workers_per_gpu % 4:
            raise ValueError("Cache worker count must divide evenly across four suites")
        shards = len(args.gpus) * args.workers_per_gpu // 4
        jobs = [(suite, shard) for shard in range(shards) for suite in ("spatial", "object", "goal", "10")]
        for index, (suite, shard) in enumerate(jobs):
            gpu = args.gpus[index % len(args.gpus)]
            data = f"/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA/libero_{suite}_no_noops_1.0.0_lerobot"
            command = [sys.executable, str(root / "repos/RoboActualizer/scripts/precompute_rgb_latents.py"),
                       "task=libero_action_only", f"encoder_tag=ctp_final_{step:06d}",
                       f"model.backbone.ckpt_path={encoder}", f"data.train.dataset_dirs=[{data}]",
                       f"data.train.rgb_latent_cache_dir={cache}", "model.latent_norm_stats_path=null",
                       f"+rgb_latent_cache_dir={cache}", "+encode_batch_size=8",
                       f"+encode_shard={shard}", f"+encode_shards={shards}", "+encode_cpu_threads=2",
                       f"+data.train.episode_ids_file={root}/experiments/libero_episode_selection.json",
                       f"hydra.run.dir={out}/cache_hydra_{suite}_{shard}"]
            log = (out / f"cache_final_{suite}_{shard}.log").open("a")
            logs.append(log)
            child = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            children.append(child)
            commands.append({"suite": suite, "shard": shard, "gpu": gpu, "pid": child.pid, "argv": command})
        status(phase="encoding", workers=commands, encoder_sha256=digest, cache=str(cache))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("Cache worker failed; inspect suite logs")
            time.sleep(10)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("Cache worker failed")
        from merge_latent_shards import merge_all
        merge_all(cache, shards, root / "experiments/libero_episode_selection.json")
        if args.rebuild_retired:
            stats = json.loads((cache / 'latent_norm_stats.json').read_text())
            if stats['encoder_checkpoint_sha256'] != digest or stats['num_files'] != 1692:
                raise RuntimeError('Preserved normalization identity mismatch')
            status(phase="complete", finished_unix=time.time(), normalization="preserved_exact_original")
            return
        status(phase="normalizing")
        subprocess.run([sys.executable, str(root / "repos/RoboActualizer/scripts/compute_latent_norm_stats.py"),
                        str(cache), "--workers", str(args.stats_workers)],
                       env={**env, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, check=True)
        status(phase="complete", finished_unix=time.time())
    except Exception as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        status(phase="failed", error=str(exc))
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()

"""Single-node video-only CTP. Run through torchrun; no SLURM/FSDP dependency."""
import argparse
import contextlib
import copy
import datetime
import fcntl
import json
import netrc
import os
import random
import signal
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from app.vjepa_2_1.transforms import make_transforms
from src.masks.multiseq_multiblock3d import MaskCollator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data import CAMERAS, DroidVideoDataset
from model import ema_update, initialize, make_backbones, Student, target_forward
from runtime import (
    OutcomeBatches, StepSchedule, atomic_save, restore_rng, rng_state,
    sha256, snapshot_link, setup_runtime_dirs,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--stage", choices=("train", "smoke", "calibrate"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--activation-checkpointing", choices=("on", "off"))
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for name in ("microbatch", "accumulation"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.activation_checkpointing is not None:
        config["activation_checkpointing"] = args.activation_checkpointing == "on"
    if args.output:
        config["output"] = str(args.output.resolve())
    if args.stage != "train" and not args.output:
        raise ValueError("Smoke/calibration must have their own explicit output directory")
    if args.stage == "calibrate" and args.resume:
        raise ValueError("Calibration cannot resume")
    if min(config["microbatch"], config["accumulation"], config["workers"]) < 1:
        raise ValueError("Invalid batch/worker configuration")
    if not 0 < config["warmup_steps"] < config["max_steps"]:
        raise ValueError("Invalid warmup schedule")
    stop = args.stop_step or config["max_steps"]
    if not 1 <= stop <= config["max_steps"]:
        raise ValueError("Invalid stop step")
    return args, config, stop


def optimizer_for(student, config):
    decay, no_decay = [], []
    for name, parameter in student.named_parameters():
        (no_decay if "bias" in name or parameter.ndim == 1 else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config["start_lr"], betas=(0.9, 0.999), eps=1e-8,
    )


def wandb_start(config, stage, metadata, resume_id):
    import wandb
    root = Path(config["root"])
    setup_runtime_dirs(root)
    for key, relative in (
        ("WANDB_DIR", "wandb"), ("WANDB_CACHE_DIR", "cache/wandb"),
        ("WANDB_CONFIG_DIR", "cache/wandb_config"), ("WANDB_DATA_DIR", "cache/wandb_data"),
    ):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    mode = os.environ.get("WANDB_MODE", config["wandb_mode"])
    if mode == "auto":
        authenticated = bool(os.environ.get("WANDB_API_KEY"))
        try:
            authenticated |= bool(netrc.netrc().authenticators("api.wandb.ai"))
        except (FileNotFoundError, netrc.NetrcParseError):
            pass
        mode = "online" if authenticated else "offline"
    return wandb.init(
        project=config["wandb_project"],
        name=config["name"] + ("" if stage == "train" else "-" + stage),
        id=resume_id, resume="must" if resume_id and mode == "online" else None,
        config=metadata, mode=mode, dir=str(root / "wandb"),
        settings=wandb.Settings(init_timeout=90),
    )


def gpu_utilization():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    return {f"gpu/{row.split(',')[0].strip()}/utilization": float(row.split(",")[1])
            for row in result.stdout.strip().splitlines()}


def resume_contract(config, manifest_hash):
    keys = (
        "seed", "world_size", "microbatch", "accumulation", "activation_checkpointing",
        "max_steps", "warmup_steps", "start_lr", "lr", "final_lr", "weight_decay",
        "ema", "encoder_source", "predictor_head_policy",
        "sampling", "lr_clock", "wall_seconds",
    )
    return {
        **{k: config[k] for k in keys}, "manifest_sha256": manifest_hash,
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("model.py", "data.py", "runtime.py", "train.py")
        },
    }


def main():
    args, config, stop_step = arguments()
    run_started_unix = time.time()
    root = Path(config["root"])
    output = root / config["output"]
    manifest = root / config["manifest"]
    output.mkdir(parents=True, exist_ok=True)
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != config["world_size"] or world != 8:
        raise ValueError("CTP-v1 requires exactly eight ranks")
    if world * config["microbatch"] * config["accumulation"] != 128:
        raise ValueError("This fixed CTP-v1 launcher uses an effective global batch of 128")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=15))
    start_clock = torch.tensor([run_started_unix], dtype=torch.float64, device=local_rank)
    dist.broadcast(start_clock, src=0)
    run_started_unix = start_clock.item()
    lock = None
    run = None
    try:
        if rank == 0:
            lock = (output / "run.lock").open("a")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (output / "latest.pt").exists() and args.resume is None:
                raise FileExistsError("Existing run: specify --resume instead of overwriting")
        dist.barrier()
        seed = config["seed"] + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        manifest_hash = sha256(manifest)
        contract = resume_contract(config, manifest_hash)
        if args.stage == "train":
            smoke = json.loads((output / "smoke/pass.json").read_text())
            calibration = json.loads((output / "calibration/selection.json").read_text())
            if not smoke["passed"] or smoke["contract"] != contract:
                raise RuntimeError("Smoke-test gate is missing or incompatible")
            for name in ("microbatch", "accumulation", "activation_checkpointing"):
                if calibration[name] != config[name]:
                    raise RuntimeError("Formal recipe does not match calibration")

        checkpoint = None
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
            if checkpoint["contract"] != contract:
                raise RuntimeError("Resume contract mismatch")
            encoder, predictor = make_backbones(config["activation_checkpointing"], predictor_output=1024)
            student = Student(encoder, predictor)
            target = copy.deepcopy(encoder)
            target.use_activation_checkpointing = False
            student.encoder.load_state_dict(checkpoint["encoder"], strict=True)
            student.predictor.load_state_dict(checkpoint["predictor"], strict=True)
            target.load_state_dict(checkpoint["ema_encoder"], strict=True)
            target.requires_grad_(False).eval()
            load_info = checkpoint["load_report"]
        else:
            student, target, load_info = initialize(config, output / f"load_report_rank{rank}.json")
        student = student.cuda(local_rank).train()
        target = target.cuda(local_rank)
        # Identical initialization on all ranks before independent masks/data RNG.
        for parameter in target.parameters():
            dist.broadcast(parameter.data, src=0)
        optimizer = optimizer_for(student, config)
        schedule = StepSchedule(optimizer, config)
        completed, resume_id = 0, None
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            schedule.load_state_dict(checkpoint["scheduler"])
            completed = checkpoint["step"]
            run_started_unix = checkpoint["run_started_unix"]
            resume_id = checkpoint["wandb_id"]
            if schedule.step != completed:
                raise RuntimeError("Scheduler/optimizer step mismatch")
        if completed >= stop_step:
            raise ValueError("Stop step must exceed the resumed step")
        ddp = DDP(student, device_ids=[local_rank], gradient_as_bucket_view=True,
                  find_unused_parameters=True, static_graph=False, broadcast_buffers=False)
        dataset = DroidVideoDataset(
            manifest, make_transforms(crop_size=256, random_resize_aspect_ratio=(0.75, 1.35),
                                      random_resize_scale=(0.3, 1.0)),
        )
        data_report = json.loads(manifest.with_name("data_report.json").read_text())
        if data_report["manifest_sha256"] != manifest_hash:
            raise RuntimeError("Frozen manifest does not match data audit")
        batch_sampler = OutcomeBatches(
            dataset.records, config["microbatch"], rank, world, config["seed"],
            completed * config["accumulation"], stop_step * config["accumulation"],
            config["accumulation"], data_report["full_episode_counts"]["1"],
            sum(data_report["full_episode_counts"].values()),
        )
        loader = DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=config["workers"],
            pin_memory=True, persistent_workers=True, prefetch_factor=config["prefetch"],
            generator=torch.Generator().manual_seed(seed),
        )
        upstream = root / "repos/vjepa2"
        recipe = yaml.safe_load((upstream / "configs/train_2_1/vitl16/pretrain-256px-16f.yaml").read_text())
        masks = MaskCollator(recipe["mask"], [16], crop_size=256, patch_size=16, tubelet_size=2)
        global_batch = config["microbatch"] * world * config["accumulation"]
        metadata = {
            **config, "stage": args.stage, "effective_global_batch": global_batch,
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
            "manifest_sha256": manifest_hash, "dataset_path": str(root / "data/droid_1.0.1"),
            "camera_views": [CAMERAS[i] for i in sorted({r["camera"] for r in dataset.records})],
            "dataset_episode_camera_samples": len(dataset),
            "dataset_completed_videos": len({r["path"] for r in dataset.records}),
            "implementation_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            "target_source": "ema_encoder", "head_adaptation": load_info["head_adaptation"],
            "actual_lr_schedule": "500-step linear warmup 2e-6->2e-5; wall-clock cosine to 2e-6 at 72h; no scaling",
            "sampling_report": data_report,
            "started_at": datetime.datetime.fromtimestamp(run_started_unix, datetime.timezone.utc).isoformat(),
            "resume_from": str(args.resume) if args.resume else None,
        }
        if rank == 0:
            (output / "resolved_config.json").write_text(json.dumps(metadata, indent=2))
            run = wandb_start(config, args.stage, metadata, resume_id)
            (output / "wandb.json").write_text(json.dumps(
                {"id": run.id, "url": run.url, "mode": run.settings.mode}, indent=2))
            print(json.dumps(metadata, indent=2), flush=True)
        iterator = iter(loader)
        if checkpoint:
            restore_rng(checkpoint["rng_per_rank"][rank])
        resumed_from = completed
        previous_sanity = checkpoint.get("sanity_passed", True) if checkpoint else True
        del checkpoint
        received_signal = [False]
        signal.signal(signal.SIGTERM, lambda *_: received_signal.__setitem__(0, True))
        signal.signal(signal.SIGINT, lambda *_: received_signal.__setitem__(0, True))
        measurements, losses = [], []
        all_sanity = previous_sanity
        wall_snapshots = {int(p.stem.split("_")[1].removesuffix("h")) for p in output.glob("wall_*h.json")}

        def save(step, final=False, wall_hour=None):
            states = [None] * world if rank == 0 else None
            dist.gather_object(rng_state(), states, dst=0)
            sanity = torch.tensor(int(all_sanity), device=local_rank)
            dist.all_reduce(sanity, op=dist.ReduceOp.MIN)
            if rank == 0:
                value = {
                    "encoder": student.encoder.state_dict(),
                    "predictor": student.predictor.state_dict(),
                    "ema_encoder": target.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                    "step": step, "clips_seen": step * global_batch,
                    "rng_per_rank": states, "contract": contract,
                    "elapsed_seconds": time.time() - run_started_unix,
                    "run_started_unix": run_started_unix,
                    "wandb_id": run.id, "load_report": load_info,
                    "metadata": metadata,
                    "sanity_passed": bool(sanity.item()),
                }
                latest = output / "latest.pt"
                atomic_save(value, latest)
                if step % config["snapshot_every"] == 0:
                    snapshot_link(latest, output / f"step_{step:06d}.pt")
                if final:
                    snapshot_link(latest, output / "final.pt")
                if final or step % config["snapshot_every"] == 0 or wall_hour is not None:
                    atomic_save(
                        {"ema_encoder": target.state_dict(),
                         "metadata": {"selection": "EMA target encoder", "step": step,
                                      "architecture": "V-JEPA 2.1 ViT-L/16",
                                      "encoder_source": "ema_encoder"}},
                        output / f"embodied_vjepa_encoder_{step:06d}.pt",
                    )
                    if wall_hour is not None:
                        snapshot_link(output / f"embodied_vjepa_encoder_{step:06d}.pt",
                                      output / f"embodied_vjepa_encoder_{wall_hour:02d}h.pt")
                        marker = output / f"wall_{wall_hour:02d}h.json"
                        temporary = marker.with_suffix(".tmp")
                        temporary.write_text(json.dumps({
                            "step": step, "elapsed_seconds": time.time() - run_started_unix,
                            "encoder": f"embodied_vjepa_encoder_{wall_hour:02d}h.pt",
                            "selection": "EMA target encoder",
                        }, indent=2))
                        temporary.replace(marker)
            dist.barrier()

        if time.time() - run_started_unix >= config["wall_seconds"]:
            save(completed, final=True)
            if rank == 0:
                print("Wall-clock cap already reached; saved final without an extra optimizer step.", flush=True)
            return
        for step in range(completed + 1, stop_step + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            # One clock for all ranks: differing LR would desynchronize DDP weights.
            elapsed_clock = torch.tensor([time.time() - run_started_unix],
                                         dtype=torch.float64, device=local_rank)
            dist.broadcast(elapsed_clock, src=0)
            lr = schedule.advance(step, elapsed=elapsed_clock.item())
            loss_sum = torch.zeros((), device=local_rank)
            components_sum = torch.zeros(2, 2, device=local_rank)
            data_wait, compute_events = 0.0, []
            for micro in range(config["accumulation"]):
                wait_start = time.perf_counter()
                data = next(iterator)
                clips = data[0][0].cuda(local_rank, non_blocking=True)
                global_micro = (step - 1) * config["accumulation"] + micro
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed + global_micro)
                    enc_masks, pred_masks = [], []
                    for generator in masks.mask_generators[16]:
                        generator._itr_counter.value = global_micro - 1
                        enc, pred = generator(config["microbatch"])
                        enc_masks.append(enc.cuda(local_rank, non_blocking=True))
                        pred_masks.append(pred.cuda(local_rank, non_blocking=True))
                data_wait += time.perf_counter() - wait_start
                begin_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin_event.record()
                synchronize = micro == config["accumulation"] - 1
                with contextlib.nullcontext() if synchronize else ddp.no_sync():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = target_forward(target, clips)
                        loss, components = ddp(clips, h, enc_masks, pred_masks, step - 1)
                    (loss / config["accumulation"]).backward()
                loss_sum += loss.detach() / config["accumulation"]
                components_sum += components / config["accumulation"]
                end_event.record()
                compute_events.append((begin_event, end_event))
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Nonfinite JEPA loss")
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float("inf"), error_if_nonfinite=True)
            encoder_grad = student.encoder.patch_embed.proj.weight.grad
            if encoder_grad is None or not torch.isfinite(encoder_grad).all() or encoder_grad.norm() == 0:
                raise RuntimeError("Online encoder has no finite nonzero gradient")
            if any(p.grad is not None or p.requires_grad for p in target.parameters()):
                raise RuntimeError("Target encoder unexpectedly has gradients")
            probe = target.patch_embed.proj.weight.flatten()[:1024].clone()
            opt_start, opt_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            opt_start.record()
            optimizer.step()
            ema_update(student.encoder, target, config["ema"])
            opt_end.record()
            torch.cuda.synchronize()
            ema_delta = (target.patch_embed.proj.weight.flatten()[:1024] - probe).abs().max()
            all_sanity &= bool(ema_delta > 0)
            compute_ms = sum(a.elapsed_time(b) for a, b in compute_events)
            elapsed = time.perf_counter() - step_start
            timing = torch.tensor(
                [elapsed, data_wait, compute_ms / 1000, opt_start.elapsed_time(opt_end) / 1000,
                 torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                dtype=torch.float64, device=local_rank,
            )
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
            dist.all_reduce(loss_sum)
            dist.all_reduce(components_sum)
            loss_value = (loss_sum / world).item()
            components_cpu = (components_sum / world).cpu().tolist()
            measurements.append(timing[0].item())
            losses.append(loss_value)
            metrics = {
                "train/loss": loss_value, "train/lr": lr,
                "train/weight_decay": config["weight_decay"], "train/grad_norm": grad_norm.item(),
                "train/ema": config["ema"], "train/ema_max_delta": ema_delta.item(),
                "train/encoder_grad_norm": encoder_grad.norm().item(),
                "optimizer_step": step, "clips_seen": step * global_batch,
                "data/success_clips": batch_sampler.successes(step - 1),
                "data/failure_clips": global_batch - batch_sampler.successes(step - 1),
                "data/target_success_fraction": data_report["success_fraction"],
                "train/context_weight": student.context_schedule.value(step - 1),
                "elapsed_hours": (time.time() - run_started_unix) / 3600,
                "perf/step_seconds": timing[0].item(),
                "perf/samples_per_second": global_batch / timing[0].item(),
                "perf/data_and_mask_wait_seconds": timing[1].item(),
                "perf/forward_backward_including_allreduce_seconds": timing[2].item(),
                "perf/optimizer_ema_seconds": timing[3].item(),
                "perf/gpu_allocated_max_bytes": timing[4].item(),
                "perf/gpu_reserved_max_bytes": timing[5].item(),
                "perf/gpu_allocated_bytes": torch.cuda.memory_allocated(),
                "perf/gpu_reserved_bytes": torch.cuda.memory_reserved(),
                **{f"train/mask_{i}/{name}": components_cpu[i][j]
                   for i in range(2) for j, name in enumerate(("prediction", "context"))},
            }
            if rank == 0:
                if step % config["log_every"] == 0 or step == resumed_from + 1:
                    metrics.update(gpu_utilization())
                    print(json.dumps(metrics), flush=True)
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                run.log(metrics, step=step)
            elapsed_wall = time.time() - run_started_unix
            low_disk = rank == 0 and step % config["latest_every"] == 0 and shutil.disk_usage(output).free < 12 * 1024**3
            stop = torch.tensor(
                [received_signal[0] or elapsed_wall >= config["wall_seconds"] or low_disk],
                device=local_rank, dtype=torch.int32,
            )
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            last = step == stop_step or bool(stop.item())
            hour = torch.tensor([next((h for h in (24, 48, 72)
                                if h not in wall_snapshots and elapsed_wall >= h * 3600), 0)
                                 if rank == 0 else 0], device=local_rank)
            dist.broadcast(hour, src=0)
            wall_hour = int(hour.item()) or None
            if wall_hour is not None:
                wall_snapshots.add(wall_hour)
            if args.stage != "calibrate" and (last or wall_hour or step % config["latest_every"] == 0):
                save(step, final=last, wall_hour=wall_hour)
            if low_disk:
                print("Disk safety stop: final checkpoint saved, less than 12 GiB free.", flush=True)
            completed = step
            if last:
                break
        passed = torch.tensor(int(all_sanity), device=local_rank)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        if rank == 0:
            steady = measurements[5:] or measurements
            summary = {
                "stage": args.stage, "step": completed, "resumed_from": resumed_from,
                "sanity_passed_all_ranks": bool(passed.item()),
                "loss_first": losses[0], "loss_last": losses[-1],
                "loss_min": min(losses), "loss_max": max(losses),
                "mean_step_seconds": float(np.mean(steady)),
                "samples_per_second": global_batch / float(np.mean(steady)),
                "physical_batch_per_gpu": config["microbatch"],
                "accumulation": config["accumulation"],
                "effective_global_batch": global_batch,
                "activation_checkpointing": config["activation_checkpointing"],
                "max_allocated_bytes": timing[4].item(),
                "max_reserved_bytes": timing[5].item(),
                "device_total_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
                "expected_steps_72h": min(config["max_steps"], int(config["wall_seconds"] / np.mean(steady))),
                "contract": contract, "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            if args.stage == "smoke" and completed == 100 and resumed_from > 0:
                summary["passed"] = bool(passed.item()) and max(losses) > min(losses)
                (output / "pass.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2), flush=True)
        dist.barrier()
    finally:
        if run:
            run.finish()
        if lock:
            lock.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

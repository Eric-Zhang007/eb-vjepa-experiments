"""Schedules, deterministic sampling, and atomic resumable checkpoints."""
import hashlib
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


def setup_runtime_dirs(root):
    root = Path(root)
    for key, subdirectory in (
        ("TMPDIR", "cache/tmp"), ("CUDA_CACHE_PATH", "cache/cuda"),
        ("TORCH_HOME", "cache/torch"), ("TRITON_CACHE_DIR", "cache/triton"),
        ("XDG_CACHE_HOME", "cache/xdg"),
    ):
        directory = root / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    tempfile.tempdir = os.environ["TMPDIR"]


def lr_at(step, config):
    """One-based optimizer step; no batch-size LR scaling."""
    if not 1 <= step <= config["max_steps"]:
        raise ValueError(f"Optimizer step outside schedule: {step}")
    warmup = config["warmup_steps"]
    if step <= warmup:
        return config["start_lr"] + (config["lr"] - config["start_lr"]) * step / warmup
    progress = (step - warmup) / (config["max_steps"] - warmup)
    return config["final_lr"] + (config["lr"] - config["final_lr"]) * (1 + math.cos(math.pi * progress)) / 2


class StepSchedule:
    def __init__(self, optimizer, config):
        self.optimizer, self.config, self.step = optimizer, config, 0

    def advance(self, step, elapsed=None):
        if step != self.step + 1:
            raise ValueError("Nonconsecutive optimizer schedule step")
        if self.config.get("lr_clock") == "wall":
            if elapsed is None:
                raise ValueError("Wall-clock schedule requires elapsed seconds")
            if step <= self.config["warmup_steps"]:
                lr = lr_at(step, self.config)
                self.warmup_elapsed = elapsed
            else:
                horizon = self.config["wall_seconds"] - self.warmup_elapsed
                progress = min(1.0, max(0.0, (elapsed - self.warmup_elapsed) / horizon))
                lr = self.config["final_lr"] + (self.config["lr"] - self.config["final_lr"]) * (
                    1 + math.cos(math.pi * progress)) / 2
        else:
            lr = lr_at(step, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step = step
        return lr

    def state_dict(self):
        return {"step": self.step, "warmup_elapsed": getattr(self, "warmup_elapsed", 0),
                "lr": self.optimizer.param_groups[0]["lr"]}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.warmup_elapsed = state.get("warmup_elapsed", 0)
        if self.step:
            for group in self.optimizer.param_groups:
                group["lr"] = state.get("lr", lr_at(self.step, self.config))


class OutcomeBatches(Sampler):
    """Proportional global batches, deterministic class shuffles and exact resume."""

    def __init__(self, records, batch, rank, world, seed, start_micro, end_micro,
                 accumulation, success_count, episode_count):
        self.pools = [[i for i, r in enumerate(records) if r["outcome"] == cls]
                      for cls in (0, 1)]
        self.batch, self.rank, self.world, self.seed = batch, rank, world, seed
        self.start, self.end, self.accumulation = start_micro, end_micro, accumulation
        self.global_batch = batch * world * accumulation
        self.success_count, self.episode_count = success_count, episode_count
        if any(len(pool) < self.global_batch for pool in self.pools):
            raise ValueError("Outcome pool too small")
        if not 0 < success_count < episode_count:
            raise ValueError("Expected both real outcome classes")
        self.orders = {}

    def __len__(self):
        return self.end - self.start

    def cumulative_success(self, steps):
        return (steps * self.global_batch * self.success_count + self.episode_count // 2) // self.episode_count

    def successes(self, step_zero_based):
        return self.cumulative_success(step_zero_based + 1) - self.cumulative_success(step_zero_based)

    def _draw(self, cls, start, count):
        pool = self.pools[cls]
        result = []
        while count:
            cycle, offset = divmod(start, len(pool))
            if self.orders.get(cls, (None,))[0] != cycle:
                generator = torch.Generator().manual_seed(self.seed + cls * 1000003 + cycle)
                self.orders[cls] = cycle, torch.randperm(len(pool), generator=generator).tolist()
            take = min(count, len(pool) - offset)
            result.extend(pool[j] for j in self.orders[cls][1][offset:offset + take])
            count -= take
            start += take
        return result

    def __iter__(self):
        last_step, indices = None, None
        for micro in range(self.start, self.end):
            step, sub = divmod(micro, self.accumulation)
            if step != last_step:
                success_start = self.cumulative_success(step)
                ns = self.successes(step)
                indices = self._draw(1, success_start, ns) + self._draw(
                    0, step * self.global_batch - success_start, self.global_batch - ns)
                generator = torch.Generator().manual_seed(self.seed + step + 2000003)
                order = torch.randperm(self.global_batch, generator=generator).tolist()
                indices = [indices[j] for j in order]
                last_step = step
            begin = (sub * self.world + self.rank) * self.batch
            yield [(index, (self.seed + step * self.global_batch + begin + j) % (2**32))
                   for j, index in enumerate(indices[begin:begin + self.batch])]


class ResumableBatches(Sampler):
    """Rebuild the exact global sample stream from the next microstep."""
    def __init__(self, size, batch, rank, world, seed, start_micro, end_micro):
        self.size, self.batch, self.rank, self.world = size, batch, rank, world
        self.seed, self.start, self.end = seed, start_micro, end_micro
        self.per_epoch = size // (batch * world)
        if self.per_epoch < 1:
            raise ValueError("Dataset is smaller than a physical global batch")

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        previous_epoch, order = None, None
        for micro in range(self.start, self.end):
            epoch, position = divmod(micro, self.per_epoch)
            if epoch != previous_epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.size, generator=generator).tolist()
                previous_epoch = epoch
            begin = (position * self.world + self.rank) * self.batch
            yield [(index, (self.seed + epoch * self.size + index) % (2**32))
                   for index in order[begin:begin + self.batch]]


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def snapshot_link(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".linktmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

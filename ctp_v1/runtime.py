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

    def advance(self, step):
        if step != self.step + 1:
            raise ValueError("Nonconsecutive optimizer schedule step")
        lr = lr_at(step, self.config)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step = step
        return lr

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]
        if self.step:
            for group in self.optimizer.param_groups:
                group["lr"] = lr_at(self.step, self.config)


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

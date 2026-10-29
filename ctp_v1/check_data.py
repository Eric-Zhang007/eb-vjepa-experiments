"""Bounded PyAV + official augmentation/masking integration check."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from app.vjepa_2_1.transforms import make_transforms
from src.masks.multiseq_multiblock3d import MaskCollator
from torch.utils.data import DataLoader

from data import DroidVideoDataset, build_episode_manifest, decode_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    random.seed(239)
    np.random.seed(239)
    torch.manual_seed(239)
    torch.set_num_threads(1)
    output = args.root / "experiments/ctp_v1"
    manifest = output / "episodes_available.jsonl"
    summary = build_episode_manifest(
        args.root / "data/droid_1.0.1",
        args.root / "data/manifests/droid_all.csv", manifest,
    )
    with open(args.root / "repos/vjepa2/configs/train_2_1/vitl16/pretrain-256px-16f.yaml") as handle:
        config = yaml.safe_load(handle)
    transform = make_transforms(
        crop_size=256, random_resize_aspect_ratio=(0.75, 1.35),
        random_resize_scale=(0.3, 1.0),
    )
    dataset = DroidVideoDataset(manifest, transform)
    examples = []
    for index in np.linspace(0, len(dataset) - 1, 4).astype(int):
        record = dataset.records[index]
        start = (record["start"] + record["end"] - 15 / 4 - 1 / 15) / 2
        frames, timestamps = decode_clip(record["path"], start, record["end"])
        assert frames.shape == (16, 180, 320, 3), frames.shape
        assert timestamps[0] >= record["start"] - 1e-6
        assert timestamps[-1] < record["end"]
        examples.append({
            **record, "rgb_shape": list(frames.shape),
            "timestamps": timestamps.tolist(),
        })
    collator = MaskCollator(
        config["mask"], [16], crop_size=256, patch_size=16, tubelet_size=2,
    )
    loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=2,
        persistent_workers=True, pin_memory=True, prefetch_factor=2,
        collate_fn=collator,
    )
    timings = []
    iterator = iter(loader)
    for _ in range(4):
        start = time.perf_counter()
        sample = next(iterator)
        timings.append(time.perf_counter() - start)
        assert len(sample) == 1
        data, enc, pred = sample[0]
        clips = data[0][0]
        assert clips.shape == (2, 3, 16, 256, 256), clips.shape
        assert torch.isfinite(clips).all()
        assert len(enc) == len(pred) == 2
        for e, p in zip(enc, pred):
            assert e.shape[0] == p.shape[0] == 2
            assert e.min() >= 0 and p.min() >= 0
            assert e.max() < 2048 and p.max() < 2048
    summary.update({
        "passed": True, "examples": examples, "loader_wait_seconds": timings,
        "batches": 4, "batch_size": 2, "decoder": "PyAV rgb24",
        "training_smoke": "NOT RUN", "throughput_calibration": "NOT RUN",
    })
    report = output / "data_smoke.json"
    report.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

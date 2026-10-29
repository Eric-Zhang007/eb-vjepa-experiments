#!/usr/bin/env python3
"""Smoke-test DROID video manifests with the official V-JEPA VideoDataset.

Run (any cwd):
  /data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/python \
      /data/users/jiachangzhang/eb_vjepa/scripts/check_dataloader.py

Options: --csv PATH --fpc 16 --fps 4 --n 2
"""
import argparse
import os
import sys

REPO = "/data/users/jiachangzhang/eb_vjepa/repos/vjepa2"
DEFAULT_CSV = "/data/users/jiachangzhang/eb_vjepa/data/manifests/droid_all.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--fpc", type=int, default=16)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    from src.datasets.video_dataset import VideoDataset

    ds = VideoDataset(
        data_paths=[args.csv],
        frames_per_clip=args.fpc,
        dataset_fpcs=[args.fpc],
        fps=args.fps,
        frame_step=None,
    )
    total = len(ds)
    existing = [i for i, s in enumerate(ds.samples) if isinstance(s, str) and os.path.exists(s)]
    print("csv:", args.csv)
    print("samples:", total, "| files existing:", len(existing))
    if not existing:
        print("NOTE: no complete video files yet (download in progress) - construction OK")
        return
    for k in existing[: args.n]:
        out = ds[k]
        if out is None:
            print(k, "load FAILED")
            continue
        buffer, label, clip_indices = out
        if isinstance(buffer, list):
            shapes = [tuple(c.shape) for c in buffer]
        else:
            shapes = [tuple(buffer.shape)]
        print("idx", k, "| label", label, "| clips", shapes, "| indices_len", len(clip_indices))
    print("OK")


if __name__ == "__main__":
    main()

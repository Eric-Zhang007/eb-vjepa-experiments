#!/usr/bin/env python3
"""Inspect a V-JEPA checkpoint: top-level keys, subtree shapes, block counts.

Usage: python3 inspect_ckpt.py [path]
"""
import sys
from collections import Counter

import torch

DEFAULT = "/data/users/jiachangzhang/eb_vjepa/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def block_stats(sd, prefix=""):
    """Count blocks like <prefix>.blocks.<i>. and total tensor params."""
    ids = set()
    params = 0
    n_tensors = 0
    for k, v in sd.items():
        if hasattr(v, "numel"):
            n_tensors += 1
            params += v.numel()
        if ".blocks." in k:
            frag = k.split(".blocks.", 1)[1]
            try:
                ids.add(int(frag.split(".", 1)[0]))
            except Exception:
                pass
    out = {"tensors": n_tensors, "params_M": round(params / 1e6, 1)}
    if ids:
        out["blocks"] = max(ids) + 1
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print("PATH:", path)
    ck = load(path)
    print("TYPE:", type(ck).__name__)
    if not isinstance(ck, dict):
        print("Not a dict; abort generic inspection.")
        return
    tc = Counter(type(v).__name__ for v in ck.values())
    print("TOP_KEYS:", list(ck.keys()))
    print("TOP_TYPES:", dict(tc))
    for k, v in ck.items():
        if isinstance(v, dict):
            st = block_stats(v)
            print("SUBTREE", k, st)
            # a couple of sample keys
            sample = list(v.keys())[:4]
            print("   sample:", sample)
        else:
            t = type(v).__name__
            extra = ""
            if hasattr(v, "shape"):
                extra = " shape=" + str(tuple(v.shape))
            print("SCALAR", k, t + extra)


if __name__ == "__main__":
    main()

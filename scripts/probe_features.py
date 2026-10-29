#!/usr/bin/env python3
"""Feature-health probe: released 0k ema_encoder vs a CTP export snapshot, on fixed clips.

Usage:
  envs/eb_vjepa_pretrain/bin/python scripts/probe_features.py --ckpt <export.pt> --tag 010000 [--train_n 2] [--held_n 2]
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data/users/jiachangzhang/eb_vjepa/ctp_v1")
sys.path.insert(0, "/data/users/jiachangzhang/eb_vjepa/repos/vjepa2")
torch.set_num_threads(128)

from model import make_backbones, clean_state  # noqa: E402
from data import decode_clip  # noqa: E402

ROOT = "/data/users/jiachangzhang/eb_vjepa"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def prep(frames):
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().numpy()
    arr = np.asarray(frames)
    if arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    else:
        x = arr.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
    t = torch.from_numpy(x)
    if t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    t = (t - MEAN) / STD
    return t.permute(1, 0, 2, 3).unsqueeze(0)


def feats(enc, x):
    with torch.no_grad():
        out = enc(x)
    while isinstance(out, (list, tuple)):
        pick = None
        for item in out:
            if isinstance(item, torch.Tensor) and item.dim() in (2, 3):
                pick = item
        out = pick if pick is not None else out[0]
    if out.dim() == 3:
        out = out[0]
    return out.float()


def fstats(f):
    n = f.norm(dim=1)
    fn = f / (n[:, None] + 1e-8)
    sim = fn @ fn.t()
    off = sim[~torch.eye(sim.shape[0], dtype=bool)]
    m = f - f.mean(0, keepdim=True)
    s = torch.linalg.svdvals(m)
    pr = float((s.sum() ** 2) / (s ** 2).sum())
    return {"norm_mean": round(float(n.mean()), 3), "norm_std": round(float(n.std()), 3),
            "dim_std_mean": round(float(f.std(0).mean()), 4),
            "pair_cos": round(float(off.mean()), 4), "eff_rank": round(pr, 1)}


def load_enc(state):
    enc, _ = make_backbones(False, predictor_output=1664)
    enc.load_state_dict(clean_state(state), strict=True)
    enc.eval()
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="snapshot")
    ap.add_argument("--train_n", type=int, default=2)
    ap.add_argument("--held_n", type=int, default=2)
    args = ap.parse_args()

    base = torch.load(ROOT + "/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt",
                      map_location="cpu", weights_only=False, mmap=True)
    exp = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    print("tag:", args.tag, "| export_metadata:", json.dumps(exp.get("metadata", {}))[:200])
    enc0 = load_enc(base["ema_encoder"])
    encn = load_enc(exp["ema_encoder"])
    clips = json.load(open(ROOT + "/experiments/ctp_v1/diagnostics_005000/fixed_clips.json"))
    sel = [c for c in clips if c.get("split") == "train"][: args.train_n] + [c for c in clips if c.get("split") == "heldout"][: args.held_n]
    deltas = []
    for c in sel:
        r = decode_clip(c["path"], c["clip_start"], c["clip_start"] + 4.0)
        frames = r[0] if isinstance(r, tuple) else r
        x = prep(frames)
        f0 = feats(enc0, x)
        f5 = feats(encn, x)
        cos = float((F.normalize(f0, dim=1) * F.normalize(f5, dim=1)).sum(1).mean())
        s0, s5 = fstats(f0), fstats(f5)
        print("clip ep", c["episode"], c["split"], "| 0k:", s0, "| ", args.tag + ":", s5, "| cos(0k,new):", round(cos, 4))


main()

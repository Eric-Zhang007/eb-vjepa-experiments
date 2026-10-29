#!/usr/bin/env python3
"""Decode smoke test for DROID AV1 mp4s using PyAV (decord is incompatible with AV1).

Usage: envs/eb_vjepa_pretrain/bin/python scripts/check_decoder.py [--n 6] [--min-age 30]
"""
import argparse
import os
import random
import time

import av

DEFAULT_CSV = "/data/users/jiachangzhang/eb_vjepa/data/manifests/droid_all.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--min-age", type=int, default=30, help="only test files older than N minutes (stable, fully downloaded)")
    args = ap.parse_args()

    rows = [l.split() for l in open(args.csv) if l.strip()]
    paths = [r[0] for r in rows]
    now = time.time()
    cands = [p for p in paths if os.path.exists(p) and (now - os.path.getmtime(p)) > args.min_age * 60]
    print("csv:", args.csv, "| total:", len(paths), "| stable candidates:", len(cands))
    random.seed(0)
    sample = random.sample(cands, min(args.n, len(cands)))
    ok = 0
    for p in sample:
        try:
            c = av.open(p)
            s = c.streams.video[0]
            shapes = [f.to_ndarray().shape for i, f in zip(range(2), c.decode(s))]
            print("OK", s.codec_context.name, shapes, p)
            ok += 1
        except Exception as e:
            print("FAIL", repr(e)[:140], p)
    print("decoded_ok %d/%d" % (ok, len(sample)))


if __name__ == "__main__":
    main()

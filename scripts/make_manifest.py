#!/usr/bin/env python3
"""Build V-JEPA-compatible video manifests for the DROID (lerobot v3.0) tree.

CSV format expected by src/datasets/video_dataset.py:
    <path> <label>   -- space-delimited, no header (column0=path, column1=label)

Outputs (under eb_vjepa/data/manifests/):
    droid_all.csv                  -- all cameras, label = camera index
    droid_<camera>.csv             -- per-camera
    droid_summary.json             -- counts/sizes per camera
"""
import json
from pathlib import Path

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
DATA = ROOT / "data" / "droid_1.0.1"
OUT = ROOT / "data" / "manifests"
CAMS = ["exterior_1_left", "exterior_2_left", "wrist_left"]


def videos_for(cam):
    d = DATA / "videos" / ("observation.images." + cam) / "chunk-000"
    return sorted(d.glob("file-*.mp4"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {"root": str(DATA), "cameras": {}}
    for ci, cam in enumerate(CAMS):
        vids = videos_for(cam)
        rows = [(str(v), ci) for v in vids]
        with open(OUT / ("droid_" + cam + ".csv"), "w") as f:
            for p, l in rows:
                f.write(p + " " + str(l) + "\n")
        all_rows += rows
        summary["cameras"][cam] = {
            "files": len(vids),
            "bytes": sum(v.stat().st_size for v in vids),
        }
    with open(OUT / "droid_all.csv", "w") as f:
        for p, l in all_rows:
            f.write(p + " " + str(l) + "\n")
    summary["total_files"] = len(all_rows)
    summary["total_bytes"] = sum(s["bytes"] for s in summary["cameras"].values())
    (OUT / "droid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

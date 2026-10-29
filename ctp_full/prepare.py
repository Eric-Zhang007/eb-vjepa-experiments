"""Freeze full DROID video inventory and audit the explicit outcome field."""
import collections
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from data import CAMERAS, build_episode_manifest
from runtime import sha256

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
out = ROOT / "experiments/ctp_full_72h"
out.mkdir(parents=True, exist_ok=True)
manifest = out / "episodes.jsonl"
if manifest.exists():
    raise FileExistsError("Manifest is immutable; inspect existing preparation")
dataset = ROOT / "data/droid_1.0.1"
columns = ["episode_index", "length", "stats/is_episode_successful/min",
           "stats/is_episode_successful/max"] + [
               f"videos/{camera}/{field}" for camera in CAMERAS
               for field in ("chunk_index", "file_index")]
outcomes = {}
referenced = set()
frame_counts = collections.Counter()
for path in sorted((dataset / "meta/episodes").rglob("*.parquet")):
    for row in pq.read_table(path, columns=columns).to_pylist():
        low, high = row[columns[2]], row[columns[3]]
        if low != high or len(low) != 1 or not isinstance(low[0], bool):
            raise ValueError(f"Nonconstant or invalid outcome: {row['episode_index']}")
        if row["episode_index"] in outcomes:
            raise ValueError("Duplicate episode")
        outcomes[row["episode_index"]] = int(low[0])
        frame_counts[str(int(low[0]))] += row["length"]
        for camera in CAMERAS:
            prefix = f"videos/{camera}/"
            chunk, file = row[prefix + "chunk_index"], row[prefix + "file_index"]
            referenced.add(str(dataset / "videos" / camera /
                               f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"))
if sorted(outcomes) != list(range(95658)):
    raise ValueError("Incomplete episode metadata")
inventory = {row[0] for row in csv.reader(
    (ROOT / "data/manifests/droid_all.csv").open(), delimiter=" ") if row}
if len(inventory) != 813 or referenced - inventory:
    raise ValueError("Incomplete downloaded video inventory")
temporary = manifest.with_suffix(".preparing")
summary = build_episode_manifest(dataset, ROOT / "data/manifests/droid_all.csv", temporary)
records = [json.loads(line) for line in temporary.read_text().splitlines()]
if any(not value for value in summary["cameras"].values()):
    raise ValueError("Expected all three camera views")
with temporary.open("w") as handle:
    for record in records:
        record["outcome"] = outcomes[record["episode"]]
        handle.write(json.dumps(record) + "\n")
temporary.replace(manifest)
counts = collections.Counter(outcomes.values())
summary.update(
    outcome_field="is_episode_successful",
    outcome_metadata_columns=columns,
    full_episode_counts={str(k): v for k, v in counts.items()},
    full_frame_counts=dict(frame_counts),
    eligible_episode_camera_counts=dict(collections.Counter(
        str(outcomes[r["episode"]]) for r in records)),
    eligible_episodes=len({r["episode"] for r in records}),
    ratio_unit="episode, before minimum clip-length filtering",
    success_fraction=counts[1] / len(outcomes),
    manifest_sha256=sha256(manifest),
    labels_used_only_for_sampling=True,
    downloaded_videos=len(inventory),
    metadata_referenced_videos=len(referenced),
    unreferenced_downloaded_videos=sorted(inventory - referenced),
    excluded_short_episodes=len(outcomes) - len({r["episode"] for r in records}),
)
(out / "data_report.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

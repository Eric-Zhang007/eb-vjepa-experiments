import collections
import csv
import json
from pathlib import Path
import pyarrow.parquet as pq
from data import CAMERAS

root = Path("/data/users/jiachangzhang/eb_vjepa")
manifest = root / "experiments/ctp_full_72h/episodes.jsonl"
rows = [json.loads(line) for line in manifest.read_text().splitlines()]
inventory = {row[0] for row in csv.reader((root / "data/manifests/droid_all.csv").open(), delimiter=" ") if row}
used = {r["path"] for r in rows}
print(json.dumps({"inventory_count": len(inventory), "used_count": len(used),
                  "samples": len(rows), "cameras": dict(collections.Counter(r["camera"] for r in rows)),
                  "unused": sorted(inventory - used), "not_in_inventory": sorted(used - inventory)}, indent=2))
referenced = set()
ids = []
short = []
columns = ["episode_index", "length"] + [
    f"videos/{camera}/{field}" for camera in CAMERAS
    for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
for path in sorted((root / "data/droid_1.0.1/meta/episodes").rglob("*.parquet")):
    for row in pq.read_table(path, columns=columns).to_pylist():
        ids.append(row["episode_index"])
        if row["length"] < 58:
            short.append(row["episode_index"])
        for camera in CAMERAS:
            prefix = f"videos/{camera}/"
            chunk, file = row[prefix + "chunk_index"], row[prefix + "file_index"]
            referenced.add(str(root / "data/droid_1.0.1/videos" / camera /
                               f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"))
print(json.dumps({"metadata_count": len(ids), "contiguous_ids": sorted(ids) == list(range(95658)),
                  "metadata_referenced_video_count": len(referenced),
                  "unreferenced_downloaded_files": sorted(inventory - referenced),
                  "referenced_but_missing_files": sorted(referenced - inventory),
                  "short_episode_count": len(short)}, indent=2))

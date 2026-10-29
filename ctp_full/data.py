"""Video-only DROID adapter. Robot supervision is never loaded."""
import csv
import json
import math
import random
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

CAMERAS = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
)


def build_episode_manifest(root, video_csv, output):
    """Intersect completed-video inventory with column-projected boundaries."""
    root = Path(root).resolve()
    with open(video_csv, newline="") as handle:
        completed = {
            Path(row[0]).resolve()
            for row in csv.reader(handle, delimiter=" ")
            if row
        }
    if not completed:
        raise ValueError("Completed-video manifest is empty")
    fields = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    columns = ["episode_index"] + [
        f"videos/{camera}/{field}" for camera in CAMERAS for field in fields
    ]
    records = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path, columns=columns)
        for row in table.to_pylist():
            for camera_index, camera in enumerate(CAMERAS):
                prefix = f"videos/{camera}/"
                chunk, file_index, start, end = [
                    row[prefix + field] for field in fields
                ]
                video = root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
                if video not in completed or not video.is_file():
                    continue
                # Leave one original frame of margin after the last sample.
                if not all(math.isfinite(x) for x in (start, end)) or start < 0:
                    raise ValueError(f"Invalid boundary in {path}: {row['episode_index']}")
                if end - start < 15 / 4 + 1 / 15:
                    continue
                records.append({
                    "path": str(video), "camera": camera_index,
                    "episode": row["episode_index"], "start": start, "end": end,
                })
    if not records:
        raise ValueError("No complete episode-camera clips available")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return {
        "samples": len(records),
        "videos": len({r["path"] for r in records}),
        "cameras": {camera: sum(r["camera"] == i for r in records)
                    for i, camera in enumerate(CAMERAS)},
        "columns_read": columns,
    }


def decode_clip(path, start, end, frames=16, fps=4):
    """Seek to a keyframe and select the nearest frames on a 4Hz time grid."""
    requested = start + np.arange(frames, dtype=np.float64) / fps
    if requested[-1] >= end:
        raise ValueError("Clip crosses the episode boundary")
    decoded, timestamps = [], []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        origin = float((stream.start_time or 0) * stream.time_base)
        container.seek(
            max(0, math.floor((origin + start) / float(stream.time_base))),
            stream=stream, backward=True, any_frame=False,
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Missing frame timestamp: {path}")
            timestamp = float(frame.pts * frame.time_base) - origin
            if timestamp < start - 1e-6:
                continue
            if timestamp >= end - 1e-6:
                break
            timestamps.append(timestamp)
            decoded.append(frame.to_ndarray(format="rgb24"))
            if timestamp >= requested[-1]:
                break
    if not timestamps:
        raise RuntimeError(f"No frames decoded at {start}: {path}")
    timestamps = np.asarray(timestamps)
    indices = np.abs(timestamps[:, None] - requested[None, :]).argmin(axis=0)
    selected = timestamps[indices]
    if np.max(np.abs(selected - requested)) > 1 / 15 + 1e-4:
        raise RuntimeError(f"Missing requested frames: {path} at {start}")
    if not np.all(np.diff(selected) > 0):
        raise RuntimeError(f"Repeated or unordered frames: {path}")
    return np.stack([decoded[i] for i in indices]), selected


class DroidVideoDataset(Dataset):
    """One sample per episode and camera, compatible with official masks."""

    def __init__(self, manifest, transform):
        with open(manifest) as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError("Empty episode manifest")
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, seed = index
            state = random.getstate(), np.random.get_state(), torch.get_rng_state()
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                return self._sample(index)
            finally:
                random.setstate(state[0])
                np.random.set_state(state[1])
                torch.set_rng_state(state[2])
        return self._sample(index)

    def _sample(self, index):
        record = self.records[index]
        latest = record["end"] - 15 / 4 - 1 / 15
        start = np.random.uniform(record["start"], latest)
        frames, timestamps = decode_clip(record["path"], start, record["end"])
        return [self.transform(frames)], record["camera"], [timestamps]

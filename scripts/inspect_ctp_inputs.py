"""Read schemas and runtime facts without reading robot supervision."""
import json
from pathlib import Path

import av
import pyarrow.parquet as pq
import torch

root = Path("/data/users/jiachangzhang/eb_vjepa")
data = root / "data/droid_1.0.1"
print("torch", torch.__version__, "av", av.__version__, flush=True)
files = sorted((data / "videos").rglob("*.mp4"))
print("completed_mp4", len(files), flush=True)
print("cameras", sorted({p.parent.parent.name for p in files}), flush=True)
meta = sorted((data / "meta/episodes").rglob("*.parquet"))
schema = pq.read_schema(meta[0])
cols = [n for n in schema.names if n == "episode_index" or n.startswith("videos/")]
print("boundary_columns", cols, flush=True)
print("boundary_example", pq.read_table(meta[0], columns=cols).slice(0, 1).to_pylist(), flush=True)
with av.open(str(files[0])) as container:
    stream = container.streams.video[0]
    frame = next(container.decode(stream))
    print("rgb_shape", frame.to_ndarray(format="rgb24").shape, flush=True)
    print("stream", stream.average_rate, stream.time_base, stream.duration, flush=True)
checkpoint = root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
print("checkpoint_bytes", checkpoint.stat().st_size, flush=True)
print("gpus", torch.cuda.device_count(), flush=True)

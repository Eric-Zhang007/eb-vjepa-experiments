"""Check real video decoding and official augmentation replay by sample seed."""
from pathlib import Path

import numpy as np
import torch
from app.vjepa_2_1.transforms import make_transforms

from data import DroidVideoDataset, decode_clip

root = Path("/data/users/jiachangzhang/eb_vjepa")
torch.set_num_threads(1)
dataset = DroidVideoDataset(
    root / "experiments/ctp_v1/episodes_available.jsonl",
    make_transforms(crop_size=256, random_resize_aspect_ratio=(0.75, 1.35)),
)
first = dataset[(0, 239)]
np.random.seed(998)
torch.manual_seed(997)
second = dataset[(0, 239)]
torch.testing.assert_close(first[0][0], second[0][0], rtol=0, atol=0)
np.testing.assert_array_equal(first[2][0], second[2][0])
assert first[0][0].shape == (3, 16, 256, 256)
record = dataset.records[0]
try:
    decode_clip(record["path"], record["end"] - 1, record["end"])
except ValueError:
    pass
else:
    raise AssertionError("Cross-episode clips must be rejected")
print("PASS real DROID RGB/augmentation exact replay and episode-boundary rejection")

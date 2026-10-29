"""Validated prepared E-LAF samples and native Dream-VL image collation."""
import json
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ELAFDataset(Dataset):
    def __init__(self, manifest):
        self.path = Path(manifest)
        self.manifest = json.loads(self.path.read_text())
        for path_key, hash_key in (("stats_path", "stats_sha256"), ("source_selection", "source_selection_sha256")):
            artifact = self.path.parent / self.manifest[path_key]
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != self.manifest[hash_key]:
                raise ValueError(f"Prepared artifact identity changed: {path_key}")
        self.samples = self.manifest["samples"]
        if not self.samples:
            raise ValueError("Empty prepared dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.path.parent / self.samples[index]["path"]
        expected = self.samples[index].get("sha256")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Sample checksum mismatch: {path}")
        sample = torch.load(path, weights_only=True)
        for key, shape in {"proprio": (8,), "actions": (8, 7), "action_valid": (8,),
                           "future_latents": (16, 1024)}.items():
            if tuple(sample[key].shape) != shape:
                raise ValueError(f"Invalid {key} shape {sample[key].shape}")
        for key in ("proprio", "actions", "future_latents", "pixel_values"):
            if not torch.isfinite(sample[key]).all():
                raise ValueError(f"Nonfinite {key}")
        return sample


def collate_fn(batch, pad_token_id=0):
    length = max(len(x["input_ids"]) for x in batch)
    ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for i, sample in enumerate(batch):
        n = len(sample["input_ids"])
        ids[i, :n] = sample["input_ids"]
        attention[i, :n] = sample["attention_mask"]
    out = {"input_ids": ids, "attention_mask": attention}
    for key in ("proprio", "actions", "action_valid", "future_latents"):
        out[key] = torch.stack([sample[key] for sample in batch])
    for key in ("pixel_values", "image_grid_thw"):
        out[key] = torch.cat([sample[key] for sample in batch])
    out["provenance"] = [sample["provenance"] for sample in batch]
    return out

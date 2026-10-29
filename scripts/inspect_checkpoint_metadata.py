"""Inspect the first uncompressed data.pkl in a partial torch ZIP.

This never reads tensor storage and is NOT a checkpoint integrity/load test.
Only explicitly supported pickle globals are accepted.
"""
import collections
import json
import pickle
import struct
from pathlib import Path


def tensor(storage, offset, size, stride, *args):
    return {"shape": list(size), "dtype": storage[1], "storage": storage[2]}


class MetadataOnly(pickle.Unpickler):
    def persistent_load(self, value):
        return value

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return tensor
        if module == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"Unsupported metadata global: {module}.{name}")


root = Path("/data/users/jiachangzhang/eb_vjepa")
path = root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
with path.open("rb") as handle:
    header = handle.read(30)
    signature, version, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", header
    )
    assert signature == 0x04034B50 and method == 0, "Not an uncompressed torch ZIP"
    name = handle.read(name_len).decode()
    assert name.endswith("/data.pkl"), name
    handle.read(extra_len)
    checkpoint = MetadataOnly(handle).load()
report = {"metadata_only": True, "bytes_available": path.stat().st_size,
          "top_keys": list(checkpoint), "subtrees": {}}
for name, state in checkpoint.items():
    if isinstance(state, dict):
        tensors = {k: v for k, v in state.items()
                   if isinstance(v, dict) and "shape" in v and "dtype" in v}
        report["subtrees"][name] = tensors
output = root / "experiments/ctp_v1/checkpoint_metadata_only.json"
output.write_text(json.dumps(report, indent=2))
print("METADATA ONLY; weights/integrity NOT verified")
print("top_keys:", report["top_keys"])
for name, tensors in report["subtrees"].items():
    print(name, "tensor_count", len(tensors))
    for key, value in tensors.items():
        if any(x in key for x in ("predictor_embed", "predictor_proj", "distill", "head", "mask_tokens")):
            print(key, value["shape"])
    blocks = [k for k in tensors if "blocks." in k]
    print("block_indices", sorted({int(k.split("blocks.")[1].split(".")[0]) for k in blocks}))
print("report", output)

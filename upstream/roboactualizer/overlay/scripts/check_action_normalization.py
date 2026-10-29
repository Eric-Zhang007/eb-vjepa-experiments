"""Check real LIBERO action/state preprocessing without text, RGB, or CUDA."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import torch

from Roboactualizer.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from Roboactualizer.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
                               version_base="1.3"):
        config = compose(config_name="train", overrides=["task=libero_action_only"])
    processor = instantiate(config.data.train.processor).train()
    assert processor.norm_default_mode == "min/max" and processor.action_state_transforms is None
    stats = {"action": {"default": {}}, "state": {"default": {}}}
    files = sorted(args.dataset.glob("data/chunk-*/episode_*.parquet"))
    if not files:
        raise ValueError("No real parquet episodes found")
    frames = 0
    grippers = set()
    for path in files:
        table = pq.read_table(path, columns=["action", "observation.state"], use_threads=False)
        frames += len(table)
        for kind, column, width in (("action", "action", 7), ("state", "observation.state", 8)):
            values = torch.tensor(table[column].to_pylist(), dtype=torch.float32)
            assert values.shape == (len(table), width) and torch.isfinite(values).all()
            current = stats[kind]["default"]
            low, high = values.amin(0), values.amax(0)
            current["global_min"] = torch.minimum(current.get("global_min", low), low)
            current["global_max"] = torch.maximum(current.get("global_max", high), high)
            if kind == "action":
                grippers.update(values[:, -1].unique().tolist())
    processor.set_normalizer_from_stats(stats)
    dataset = BaseLerobotDataset(
        dataset_dirs=[str(args.dataset)],
        shape_meta=OmegaConf.to_container(config.data.train.shape_meta, resolve=True),
        action_size=16, obs_size=17, current_images_only=True,
        val_set_proportion=0, is_training_set=True)
    dataset._set_return_images(False)
    assert len(dataset) == frames
    report = {"dataset": str(args.dataset), "episodes": len(files), "frames": frames,
              "gripper_unique": sorted(grippers), "samples": [],
              "scope": "Real data processor/min-max roundtrip only; simulator action convention not established."}
    for index in (0, 6, len(dataset) - 1):
        raw = dataset[index]
        processed = processor.preprocess(deepcopy(raw))
        restored = processor.normalizer.normalizers["action"]["default"].backward(processed["action"])
        expected = raw["action"]["default"].clone()
        expected[raw["action_is_pad"], :6] = 0
        torch.testing.assert_close(restored, expected, atol=1e-6, rtol=1e-5)
        restored_state = processor.normalizer.normalizers["state"]["default"].backward(processed["proprio"])
        torch.testing.assert_close(restored_state, raw["state"]["default"], atol=1e-6, rtol=1e-5)
        assert processed["action"].shape == (16, 7)
        report["samples"].append({"index": index, "task": raw["task"],
            "max_action_error": float((restored - expected).abs().max()),
            "max_state_error": float((restored_state - raw["state"]["default"]).abs().max()),
            "padded_actions": int(raw["action_is_pad"].sum())})
    report["status"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats_path = args.output.with_name(args.output.stem + "_diagnostic_stats.json")
    save_dataset_stats_to_json(stats, str(stats_path))
    report["diagnostic_stats_path"] = str(stats_path)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

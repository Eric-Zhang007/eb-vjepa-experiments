"""Test the explicitly approved exclusion against real metadata and parquet selection."""
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from Roboactualizer.utils.config_resolvers import register_default_resolvers
from Roboactualizer.utils.episode_selection import selected_episode_ids, selection_sha256
from Roboactualizer.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    import os
    os.environ["HF_HOME"] = str(root / "cache/huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(root / "cache/huggingface/datasets")
    manifest = root / "experiments/libero_episode_selection.json"
    payload = json.loads(manifest.read_text())
    total = 0
    for name, entry in payload["datasets"].items():
        ids = selected_episode_ids(manifest, name, entry["total_episodes"])
        expected = [i for i in range(entry["total_episodes"])
                    if not (name.startswith("libero_goal_") and i == 82)]
        assert ids == expected
        total += len(ids)
    assert total == 1692
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(root / "repos/RoboActualizer/configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=libero_action_only_final"])
    data = cfg.data.train
    dataset = BaseLerobotDataset(
        dataset_dirs=list(data.dataset_dirs),
        shape_meta=OmegaConf.to_container(data.shape_meta, resolve=True),
        obs_size=17, action_size=16, current_images_only=True,
        val_set_proportion=0, is_training_set=True, strict_read=True,
        episode_ids_file=str(manifest))
    assert dataset.multi_dataset.num_episodes == 1692
    assert len(dataset) == 273465 - 129
    selected = {}
    for ds in dataset.multi_dataset._datasets:
        name = Path(ds.root).name
        ids = list(ds.episodes)
        actual = set(int(i) for i in ds.hf_dataset["episode_index"])
        assert actual == set(ids)
        if name.startswith("libero_goal_"):
            assert 82 not in actual and 81 in actual and 83 in actual
        selected[name] = len(ids)
    report = {"status": "PASS", "selection_sha256": selection_sha256(manifest),
              "episodes": 1692, "frames": len(dataset), "excluded_frames": 129,
              "suites": selected, "original_data_modified": False}
    (root / "experiments/action_only/episode_selection_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()

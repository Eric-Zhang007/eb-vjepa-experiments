"""One explicit episode selection contract for caches and training."""
import hashlib
import json
from pathlib import Path


def selection_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def selected_episode_ids(path, dataset_dir, total_episodes):
    if path is None:
        return list(range(total_episodes))
    payload = json.loads(Path(path).read_text())
    if "datasets" in payload:
        name = Path(dataset_dir).name
        if name not in payload["datasets"]:
            raise ValueError(f"Selection manifest has no entry for {name}")
        entry = payload["datasets"][name]
        if entry["total_episodes"] != total_episodes:
            raise ValueError(f"Episode count changed for {name}")
        excluded = entry["exclude_episode_ids"]
        if any(type(i) is not int or not 0 <= i < total_episodes for i in excluded):
            raise ValueError(f"Invalid excluded episode IDs for {name}")
        return [i for i in range(total_episodes) if i not in set(excluded)]
    ids = payload["episode_ids"]
    if any(type(i) is not int or not 0 <= i < total_episodes for i in ids):
        raise ValueError("Invalid selected episode IDs")
    return sorted(set(ids))

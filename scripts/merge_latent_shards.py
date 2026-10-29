"""Merge disjoint episode metadata only after complete shard coverage checks."""
import hashlib
import json


def merge_all(cache, shards, selection_path):
    selection = json.loads(selection_path.read_text())
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    for suite, spec in selection["datasets"].items():
        directory = cache / suite
        identity = json.loads((directory / "encoder_identity.json").read_text())
        expected = {str(i) for i in range(spec["total_episodes"])
                    if i not in spec["exclude_episode_ids"]}
        combined = {}
        base = None
        for shard in range(shards):
            meta = json.loads((directory / f"latents_meta_shard_{shard:03d}_of_{shards:03d}.json").read_text())
            if meta["encoder_checkpoint_sha256"] != identity["encoder_checkpoint_sha256"]:
                raise ValueError("Shard encoder identity mismatch")
            if meta["episode_selection_sha256"] != selection_sha:
                raise ValueError("Shard episode selection mismatch")
            episodes = meta.pop("episodes")
            if meta.pop("num_episodes_encoded") != len(episodes):
                raise ValueError("Shard episode count mismatch")
            if base is not None and meta != base:
                raise ValueError("Shard metadata mismatch")
            base = meta
            if combined.keys() & episodes.keys():
                raise ValueError("Duplicate episode across shards")
            combined.update(episodes)
        if set(combined) != expected:
            raise ValueError("Missing or extra episode in merged shards")
        if any(not (directory / f"episode_{int(ep):06d}.pt").is_file() for ep in combined):
            raise ValueError("Missing episode payload")
        result = {**base, "episodes": combined, "num_episodes_encoded": len(combined)}
        temp = directory / "latents_meta.tmp"
        temp.write_text(json.dumps(result, indent=2))
        temp.replace(directory / "latents_meta.json")

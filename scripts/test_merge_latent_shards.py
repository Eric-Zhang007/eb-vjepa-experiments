import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from merge_latent_shards import merge_all


class MergeTest(unittest.TestCase):
    def test_complete_and_duplicate_and_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selection = root / "selection.json"
            selection.write_text(json.dumps({"datasets": {"suite": {
                "total_episodes": 5, "exclude_episode_ids": [2]}}}))
            suite = root / "suite"
            suite.mkdir()
            (suite / "encoder_identity.json").write_text('{"encoder_checkpoint_sha256":"unit-test"}')
            meta = {"encoder_checkpoint_sha256": "unit-test",
                    "episode_selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest()}
            for shard, episodes in enumerate(({"0": 4, "3": 4}, {"1": 4, "4": 4})):
                (suite / f"latents_meta_shard_{shard:03d}_of_002.json").write_text(json.dumps({
                    **meta, "episodes": episodes, "num_episodes_encoded": 2}))
                for ep in episodes:
                    (suite / f"episode_{int(ep):06d}.pt").touch()
            merge_all(root, 2, selection)
            self.assertEqual(json.loads((suite / "latents_meta.json").read_text())["num_episodes_encoded"], 4)
            (suite / "episode_000004.pt").unlink()
            with self.assertRaisesRegex(ValueError, "payload"):
                merge_all(root, 2, selection)
            (suite / "latents_meta_shard_001_of_002.json").write_text(json.dumps({
                **meta, "episodes": {"0": 4, "4": 4}, "num_episodes_encoded": 2}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                merge_all(root, 2, selection)


if __name__ == "__main__":
    unittest.main()

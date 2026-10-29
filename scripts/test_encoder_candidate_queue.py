"""Queue safety and selection tests; all fixtures are temporary, unit-only data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from run_encoder_candidate_queue import completed_results, retire_cache
from select_action_checkpoint import select_checkpoint


class QueueTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.run = self.base / "ctp_full_040000_handoff"
        training = self.run / "train_action_only_040000"
        weights = training / "checkpoints/weights"
        weights.mkdir(parents=True)
        checkpoint = weights / "step_020000.pt"
        checkpoint.write_bytes(b"unit-only")
        self.sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (training / "checkpoint_selection.json").write_text('{"selected":{"step":20000}}')
        (self.run / "continuation_status.json").write_text('{"phase":"complete"}')
        (self.run / "formal_train_status.json").write_text('{"phase":"complete"}')
        for benchmark, n in [("clean", 200), ("plus", 280)]:
            directory = self.run / f"eval_{benchmark}_20k"
            directory.mkdir()
            (directory / "summary.json").write_text(json.dumps({
                "episodes": n, "successes": 0, "checkpoint_sha256": self.sha,
                "records": [{"suite": "unit-only", "task_id": i, "trial": 0,
                             "success": False} for i in range(n)]}))
        self.cache = self.run / "cache_retry1/cache_final_040000_exclude_goal82"
        (self.cache / "suite").mkdir(parents=True)
        (self.cache / "latent_norm_stats.json").write_text('{"num_files":1692}')
        (self.cache / "suite/encoder_identity.json").write_text('{}')
        (self.cache / "suite/episode_000000.pt").write_bytes(b"unit-only")

    def test_aggregate_and_retire_preserves_metadata(self):
        self.assertEqual(len(completed_results(self.run, 40000)), 2)
        retire_cache(self.base, 40000, True)
        self.assertFalse((self.cache / "suite/episode_000000.pt").exists())
        self.assertTrue((self.cache / "suite/encoder_identity.json").exists())
        self.assertTrue((self.cache / "latent_norm_stats.json").exists())
        self.assertTrue((self.run / "train_action_only_040000/checkpoints/weights/step_020000.pt").exists())

    def test_no_authorization_no_delete(self):
        with self.assertRaisesRegex(RuntimeError, "authorization"):
            retire_cache(self.base, 40000, False)
        self.assertTrue((self.cache / "suite/episode_000000.pt").exists())

    def test_incomplete_run_no_delete(self):
        (self.run / "continuation_status.json").write_text('{"phase":"training"}')
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            retire_cache(self.base, 40000, True)

    def test_duplicate_rollout_rejected(self):
        path = self.run / "eval_clean_20k/summary.json"
        data = json.loads(path.read_text()); data["records"][-1] = data["records"][0]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, "mismatched"):
            completed_results(self.run, 40000)

    def test_symlink_refused(self):
        outside = self.base / "outside.pt"; outside.write_bytes(b"preserve")
        (self.cache / "suite/episode_000001.pt").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            retire_cache(self.base, 40000, True)
        self.assertEqual(outside.read_bytes(), b"preserve")

    def test_selector_uses_window_not_single_minimum(self):
        training = self.run / "train_action_only_040000"
        (training / "checkpoints/weights/step_005000.pt").write_bytes(b"unit-only")
        rows = [{"step": step, "train/action_loss": (0 if step == 5000 else 0.5)} for step in range(4010, 5001, 10)]
        rows += [{"step": step, "train/action_loss": 0.2} for step in range(19010, 20001, 10)]
        metrics = training / "train_metrics.jsonl"
        metrics.write_text("\n".join(json.dumps(r) for r in rows))
        self.assertEqual(select_checkpoint(training), 20000)
        metrics.write_text("\n".join(json.dumps(r) for r in rows[:-1]))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_checkpoint(training)


if __name__ == "__main__":
    unittest.main()

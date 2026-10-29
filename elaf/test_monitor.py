"""Monitoring identity and atomic publication contracts; synthetic inputs only."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from elaf.monitor import (discover_snapshots, monitor_manifest_sha256,
                          run_monitor_snapshot, save_resampler_snapshot)


class MonitorTests(unittest.TestCase):
    def test_worker_only_consumes_completed_unlogged_snapshots(self):
        from elaf.v0_monitor import pending_snapshots
        zero = self.save(0)
        later = self.save(100)
        output = self.root / 'worker_results'
        output.mkdir()
        (output / (zero.stem + '.uploaded')).write_text('{}')
        (zero.parent / '.resampler-step-000000300.pt.tmp').write_text('partial')
        self.assertEqual(pending_snapshots(zero.parent, output), [later])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.model.weight.copy_(torch.eye(4))
        features = torch.randn(8, 3, 4, generator=torch.Generator().manual_seed(39))
        self.cache = {"schema_version": 1, "sample_ids": [f"episode-{i}:frame-8" for i in range(8)],
                      "manifest": {"encoder_checkpoint_sha256": "e" * 64, "preprocessing_sha256": "a" * 64},
                      "current_features": features, "future_features": 2 * features}
        self.sha = monitor_manifest_sha256(self.cache)

    def save(self, step):
        return save_resampler_snapshot(self.model, self.root / "snapshots", training_step=step,
            config={"latent_dim": 4}, parent_run_id="elaf-test", monitor_manifest_sha256=self.sha)

    def run_snapshot(self, snapshot, cache=None):
        return run_monitor_snapshot(self.model, cache or self.cache, snapshot, self.root / "results", batch_size=3)

    def test_step_zero_persists_and_scale_changes_relative_metric(self):
        zero = self.run_snapshot(self.save(0))
        before = (self.root / "results/baseline.json").read_bytes()
        with torch.no_grad():
            self.model.weight.mul_(.25)
        later = self.run_snapshot(self.save(100))
        self.assertAlmostEqual(zero["metrics"]["monitor/delta/rms_vs_initial"], 1)
        self.assertAlmostEqual(later["metrics"]["monitor/delta/rms_vs_initial"], .25, places=5)
        self.assertEqual(before, (self.root / "results/baseline.json").read_bytes())
        self.assertTrue((self.root / "results/diagnostics-step-000000100.pt").exists())

    def test_no_later_step_without_baseline(self):
        with self.assertRaisesRegex(ValueError, "step-zero"):
            self.run_snapshot(self.save(100))

    def test_modified_sample_features_or_encoder_rejected(self):
        snapshot = self.save(0)
        variants = []
        variants.append({**self.cache, "sample_ids": list(reversed(self.cache["sample_ids"]))})
        variants.append({**self.cache, "current_features": self.cache["current_features"] + 1})
        variants.append({**self.cache, "manifest": {**self.cache["manifest"], "encoder_checkpoint_sha256": "c" * 64}})
        for cache in variants:
            with self.subTest(cache=cache.keys()), self.assertRaisesRegex(ValueError, "identity changed"):
                self.run_snapshot(snapshot, cache)

    def test_immutable_snapshot_and_incomplete_files_ignored(self):
        snapshot = self.save(0)
        content = snapshot.read_bytes()
        with self.assertRaises(FileExistsError):
            self.save(0)
        self.assertEqual(snapshot.read_bytes(), content)
        incomplete = snapshot.parent / ".resampler-step-000000100.pt.abc.tmp"
        incomplete.write_bytes(b"unfinished")
        self.assertEqual(discover_snapshots(snapshot.parent), [snapshot])
        with self.assertRaisesRegex(ValueError, "atomically published"):
            self.run_snapshot(incomplete)

    def test_undefined_metrics_are_null_with_explicit_invalid_fields(self):
        self.cache["current_features"].zero_()
        self.cache["future_features"].zero_()
        self.sha = monitor_manifest_sha256(self.cache)
        self.run_snapshot(self.save(0))
        path = self.root / "results/diagnostics-step-000000000.json"
        report = json.loads(path.read_text())
        key = "monitor/delta/rms_vs_initial"
        self.assertIsNone(report["metrics"][key])
        self.assertIn(key, report["invalid_metric_fields"])
        self.assertNotIn("NaN", path.read_text())


if __name__ == "__main__":
    unittest.main()

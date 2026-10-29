"""Unit-only fixtures for grid selection, archive identity and gate reuse."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from grid_peak_search import decision
from grid_cache_storage import validate_archived_source
from prepare_ctp_grid import prepare


class PeakTests(unittest.TestCase):
    def test_reused_prior_10k_advances_to_13k(self):
        scores = {1: (.635 + 127 / 280) / 2,
                  9: (.735 + 144 / 280) / 2,
                  10: (.745 + 153 / 280) / 2,
                  17: (.63 + 131 / 280) / 2}
        result = decision(scores, maximum=17)
        self.assertEqual(result["next"], 13)
        self.assertEqual(result["interval"], [10, 17])

    def test_seventeen_point_grid_all_peaks(self):
        for peak in range(1, 18):
            scores = {}
            for _ in range(18):
                result = decision(scores, maximum=17)
                if result["next"] is None:
                    break
                point = result["next"]
                self.assertTrue(1 <= point <= 17)
                scores[point] = 1 - abs(point - peak) * .04
            self.assertEqual(result["best"], peak)
        self.assertEqual(decision({}, maximum=17)["next"], 9)

    def test_all_unimodal_peaks_including_boundaries(self):
        for peak in range(1, 21):
            scores = {}
            for _ in range(21):
                result = decision(scores)
                if result["next"] is None:
                    break
                point = result["next"]
                self.assertNotIn(point, scores)
                scores[point] = 1 - abs(point - peak) * 0.04
            self.assertEqual(result["best"], peak)
            for neighbor in (peak - 1, peak + 1):
                if 1 <= neighbor <= 20:
                    self.assertIn(neighbor, scores)

    def test_plateau_does_not_arbitrarily_discard_candidates(self):
        scores = {}
        for _ in range(21):
            result = decision(scores)
            if result["next"] is None:
                break
            scores[result["next"]] = 0.5
        self.assertEqual(len(scores), 20)
        self.assertTrue(result["grid_complete"])

    def test_observed_valley_switches_to_guarded_dense_search(self):
        result = decision({1: 0.8, 10: 0.1, 20: 0.9})
        self.assertEqual(result["reason"], "observed_nonunimodality")
        self.assertEqual(result["mode"], "dense_guard")

    def test_invalid_scores_rejected(self):
        for scores in ({0: 0.5}, {1: float("nan")}, {1: 1.1}):
            with self.assertRaises(ValueError):
                decision(scores)


class AuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.encoder = self.root / "embodied_vjepa_encoder_001000.pt"
        self.encoder.write_bytes(b"unit fixture only, not a model")
        self.source = self.root / "step_001000.pt"
        self.audit = self.root / "checkpoint_audit_001000.json"
        self.record = {"status": "PASS", "step": 1000, "source_checkpoint": str(self.source),
                       "source_sha256": "unit-source-sha", "encoder": str(self.encoder),
                       "encoder_sha256": hashlib.sha256(self.encoder.read_bytes()).hexdigest(),
                       "export_equals_source_ema": True, "finite": True, "tensors": 302}
        self.audit.write_text(json.dumps(self.record))
        self.receipt = self.root / "archive_receipt_001000.json"
        self.receipt.write_text(json.dumps({"phase": "complete", "step": 1000,
                                           "source_sha256": "unit-source-sha", "local_sha256_verified": True}))

    def test_archived_source_can_be_absent_when_hash_bound_receipt_exists(self):
        self.assertEqual(validate_archived_source(self.audit, self.encoder, 1000, self.source)["step"], 1000)

    def test_changed_encoder_rejected(self):
        self.encoder.write_bytes(b"different unit fixture")
        with self.assertRaises(RuntimeError):
            validate_archived_source(self.audit, self.encoder, 1000, self.source)

    def test_incomplete_receipt_rejected(self):
        self.receipt.write_text('{"phase":"verified"}')
        with self.assertRaises(RuntimeError):
            validate_archived_source(self.audit, self.encoder, 1000, self.source)

    def test_wrong_checkpoint_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_archived_source(self.audit, self.encoder, 1000, self.root / "another.pt")


class GateTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        prior = root / "experiments/ctp_full_72h"
        (prior / "smoke").mkdir(parents=True)
        (prior / "calibration").mkdir()
        (root / "ctp_full").mkdir()
        hashes = {}
        for name in ("train.py", "model.py", "data.py", "runtime.py"):
            path = root / "ctp_full" / name
            path.write_text("unit-only fixture")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = prior / "episodes.jsonl"
        manifest.write_text("unit-only manifest")
        config = {"microbatch": 16, "accumulation": 1, "activation_checkpointing": False,
                  "lr_clock": "wall", "wall_seconds": 259200, "max_steps": 1000000,
                  "warmup_steps": 500, "lr": 2e-5, "snapshot_every": 5000,
                  "manifest": str(manifest), "name": "prior", "output": str(prior)}
        contract = {k: v for k, v in config.items() if k not in ("snapshot_every", "manifest", "name", "output")}
        contract.update(implementation_sha256=hashes,
                        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        (prior / "calibrated.yaml").write_text(yaml.safe_dump(config))
        (prior / "smoke/pass.json").write_text(json.dumps({"passed": True, "sanity_passed_all_ranks": True,
                                                          "step": 100, "resumed_from": 90,
                                                          "contract": contract, "completed_at": "unit-only"}))
        (prior / "calibration/selection.json").write_text(json.dumps(config))
        return root, config

    def test_only_identity_and_snapshot_cadence_changed(self):
        root, prior = self.fixture()
        with patch("prepare_ctp_grid.shutil.disk_usage") as usage:
            usage.return_value.free = 200_000_000_000
            prepare(root)
        current = yaml.safe_load((root / "experiments/ctp_grid20k_20261008/config.yaml").read_text())
        self.assertEqual({k for k in prior if prior[k] != current[k]}, {"name", "output", "snapshot_every"})
        self.assertEqual(current["max_steps"], 1000000)
        self.assertEqual(current["wall_seconds"], 259200)
        self.assertEqual(current["snapshot_every"], 1000)

    def test_changed_implementation_cannot_reuse_gate(self):
        root, _ = self.fixture()
        (root / "ctp_full/train.py").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "Implementation changed"):
            prepare(root)


if __name__ == "__main__":
    unittest.main()

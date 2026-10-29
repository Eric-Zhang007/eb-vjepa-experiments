"""Real torch serialization on tiny CPU-only test tensors, not research results."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from archive_grid_checkpoint import inspect_checkpoint, commit


class ArchiveTests(unittest.TestCase):
    def fixture(self, step=1000):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out = Path(temp.name)
        tensors = {f"unit_{i}": torch.tensor([float(i)]) for i in range(302)}
        source = {"step": step, "sanity_passed": True, "ema_encoder": tensors,
                  "contract": {"unit_fixture": True}, "wandb_id": "unit-only"}
        torch.save(source, out / f"step_{step:06d}.pt")
        torch.save({"ema_encoder": tensors, "metadata": {"step": step}},
                   out / f"embodied_vjepa_encoder_{step:06d}.pt")
        return out

    def test_verified_commit_preserves_export_and_receipt(self):
        out = self.fixture()
        audit = inspect_checkpoint(out, 1000)
        self.assertEqual(audit["status"], "PASS")
        local = f"E:/eb_vjepa_archive/{out.name}/step_001000.pt"
        commit(out, 1000, audit["source_sha256"], local)
        self.assertFalse((out / "step_001000.pt").exists())
        self.assertTrue((out / "embodied_vjepa_encoder_001000.pt").exists())
        self.assertEqual(json.loads((out / "archive_receipt_001000.json").read_text())["phase"], "complete")
        commit(out, 1000, audit["source_sha256"], local)

    def test_20k_full_state_retained(self):
        out = self.fixture(20000)
        audit = inspect_checkpoint(out, 20000)
        commit(out, 20000, audit["source_sha256"], f"E:/eb_vjepa_archive/{out.name}/step_020000.pt")
        self.assertTrue((out / "step_020000.pt").exists())

    def test_wrong_local_digest_never_removes_source(self):
        out = self.fixture()
        with self.assertRaises(RuntimeError):
            commit(out, 1000, "wrong", f"E:/eb_vjepa_archive/{out.name}/step_001000.pt")
        self.assertTrue((out / "step_001000.pt").exists())

    def test_wrong_archive_destination_never_removes_source(self):
        out = self.fixture()
        audit = inspect_checkpoint(out, 1000)
        with self.assertRaises(RuntimeError):
            commit(out, 1000, audit["source_sha256"], "E:/elsewhere.pt")
        self.assertTrue((out / "step_001000.pt").exists())

    def test_nonfinite_export_rejected(self):
        out = self.fixture()
        path = out / "embodied_vjepa_encoder_001000.pt"
        export = torch.load(path, weights_only=False)
        export["ema_encoder"]["unit_0"] = torch.tensor([float("nan")])
        torch.save(export, path)
        with self.assertRaises(RuntimeError):
            inspect_checkpoint(out, 1000)
        self.assertFalse((out / "checkpoint_audit_001000.json").exists())

    def test_early_stop_not_reported_as_waiting_forever(self):
        out = self.fixture()
        (out / "summary.json").write_text('{"step":1000}')
        self.assertEqual(inspect_checkpoint(out, 2000)["status"], "pretraining_stopped_early")


if __name__ == "__main__":
    unittest.main()

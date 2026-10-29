"""Fast CPU regression tests; these are not the pretrained 100-step smoke."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from model import clean_state, ema_update, load_report
from runtime import ResumableBatches, StepSchedule, atomic_save, lr_at, snapshot_link

CONFIG = {
    "start_lr": 2e-6, "lr": 2e-5, "final_lr": 2e-6,
    "warmup_steps": 500, "max_steps": 30000,
}


class RuntimeTests(unittest.TestCase):
    def test_schedule_and_resume(self):
        self.assertAlmostEqual(lr_at(500, CONFIG), 2e-5)
        self.assertAlmostEqual(lr_at(30000, CONFIG), 2e-6)
        self.assertGreater(lr_at(1, CONFIG), 2e-6)
        self.assertLess(lr_at(501, CONFIG), lr_at(500, CONFIG))
        parameter = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.AdamW([parameter], lr=1)
        schedule = StepSchedule(opt, CONFIG)
        for step in range(1, 501):
            schedule.advance(step)
        restored = StepSchedule(opt, CONFIG)
        restored.load_state_dict(schedule.state_dict())
        self.assertAlmostEqual(restored.advance(501), lr_at(501, CONFIG))
        with self.assertRaises(ValueError):
            restored.advance(503)

    def test_rank_partition_and_resume(self):
        batches = [list(ResumableBatches(160, 2, rank, 8, 239, 0, 24)) for rank in range(8)]
        for micro in range(24):
            samples = [index for rank in range(8) for index, _ in batches[rank][micro]]
            self.assertEqual(len(samples), len(set(samples)))
        tail = list(ResumableBatches(160, 2, 3, 8, 239, 13, 24))
        self.assertEqual(tail, batches[3][13:])
        with self.assertRaises(ValueError):
            ResumableBatches(3, 2, 0, 8, 239, 0, 1)

    def test_strict_report_and_prefixes(self):
        module = torch.nn.Linear(3, 2)
        state = clean_state({f"module.backbone.{k}": v for k, v in module.state_dict().items()})
        report = load_report(module, state)
        self.assertEqual(report["loaded_numel_pct"], 100)
        self.assertEqual(report["missing_keys"], [])
        state["weight"] = torch.ones(5, 3)
        state["unexpected"] = torch.ones(1)
        del state["bias"]
        report = load_report(module, state)
        self.assertEqual(report["missing_keys"], ["bias"])
        self.assertEqual(report["unexpected_keys"], ["unexpected"])
        self.assertIn("weight", report["shape_mismatches"])
        with self.assertRaises(ValueError):
            clean_state({"weight": torch.ones(1), "module.weight": torch.ones(1)})

    def test_ema_has_no_grad(self):
        online = torch.nn.Linear(3, 2)
        target = copy.deepcopy(online).requires_grad_(False)
        before = target.weight.clone()
        with torch.no_grad():
            online.weight.add_(2)
        ema_update(online, target, 0.9)
        torch.testing.assert_close(target.weight, before + 0.2)
        self.assertIsNone(target.weight.grad)
        self.assertFalse(target.weight.requires_grad)

    def test_atomic_checkpoint_snapshot_and_optimizer_resume(self):
        torch.manual_seed(239)
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        x = torch.randn(2, 3)
        model(x).square().mean().backward()
        opt.step()
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / "latest.pt"
            snapshot = Path(directory) / "step_1.pt"
            atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict()}, latest)
            snapshot_link(latest, snapshot)
            restored = torch.load(latest, weights_only=False)
            replica = torch.nn.Linear(3, 2)
            replica.load_state_dict(restored["model"], strict=True)
            replica_opt = torch.optim.AdamW(replica.parameters(), lr=2e-5)
            replica_opt.load_state_dict(restored["optimizer"])
            for net, optimizer in ((model, opt), (replica, replica_opt)):
                optimizer.zero_grad()
                net(x).square().mean().backward()
                optimizer.step()
            torch.testing.assert_close(model.weight, replica.weight, rtol=0, atol=0)
            atomic_save({"replacement": True}, latest)
            self.assertIn("model", torch.load(snapshot, weights_only=False))
            self.assertIn("replacement", torch.load(latest, weights_only=False))


if __name__ == "__main__":
    unittest.main(verbosity=2)

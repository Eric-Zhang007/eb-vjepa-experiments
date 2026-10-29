"""Synthetic mathematical contracts for standard representation diagnostics."""
import math
import unittest
import torch
from elaf.diagnostics import representation_diagnostics, slot_linear_cka


class DiagnosticsTests(unittest.TestCase):
    def test_constant_slots_do_not_fake_sample_diversity(self):
        z = torch.arange(12).reshape(1, 3, 4).expand(8, -1, -1).float()
        result = representation_diagnostics(z, z)
        self.assertEqual(result["metrics"]["monitor/z/global_erank"], 0)
        self.assertTrue((result["current"]["slot_variance"] == 0).all())
        self.assertTrue((result["current"]["slot_erank"] == 0).all())
        self.assertTrue(torch.isnan(result["slot_cka"]).all())
        self.assertEqual(result["metrics"]["monitor/delta/rms_raw"], 0)
        self.assertTrue(math.isnan(result["metrics"]["monitor/delta/rms_vs_initial"]))

    def test_redundant_slots_and_invalid_slot(self):
        x = torch.randn(20, 4, generator=torch.Generator().manual_seed(12))
        z = torch.stack([x, 3 * x + 7, torch.ones_like(x)], dim=1)
        cka, valid = slot_linear_cka(z)
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertAlmostEqual(cka[0, 1].item(), 1, places=5)
        self.assertTrue(torch.isnan(cka[2]).all())

    def test_nonbinary_constant_and_absolute_variance_floor(self):
        z = torch.full((19, 2, 7), .1)
        result = representation_diagnostics(z, z)
        self.assertEqual(result["metrics"]["monitor/z/global_erank"], 0)
        self.assertFalse(result["cka_valid_slots"].any())
        z[:, 0, :] += torch.arange(19).float()[:, None] * 1e-6
        _, valid = slot_linear_cka(z, variance_floor=1e-8)
        self.assertFalse(valid.any())

    def test_structural_rank_cap_does_not_change_global_or_original_norm(self):
        a = torch.tensor([1., -1., 1., -1.])
        b = torch.tensor([1., 1., -1., -1.])
        z = torch.stack([a, b], dim=1).unsqueeze(-1).expand(-1, -1, 5)
        result = representation_diagnostics(z, z, representation_rank_cap=1)
        torch.testing.assert_close(result["current"]["slot_erank_structural_norm"], torch.ones(2))
        torch.testing.assert_close(result["current"]["slot_erank_norm"], torch.full((2,), 1 / 3))
        self.assertAlmostEqual(result["metrics"]["monitor/z/global_erank"], 2, places=5)

    def test_equal_energy_orthogonal_rank_and_cka(self):
        # Two centered orthogonal sample directions, each rank one.
        a = torch.tensor([1., -1., 1., -1.])
        b = torch.tensor([1., 1., -1., -1.])
        z = torch.stack([a, b], dim=1).unsqueeze(-1)
        result = representation_diagnostics(z, z + 1)
        torch.testing.assert_close(result["current"]["slot_erank"], torch.ones(2))
        self.assertAlmostEqual(result["metrics"]["monitor/z/global_erank"], 2, places=5)
        self.assertAlmostEqual(result["slot_cka"][0, 1].item(), 0, places=5)
        self.assertEqual(result["metrics"]["monitor/delta/global_erank"], 0)

    def test_scale_changes_raw_delta_but_not_rank(self):
        z = torch.randn(12, 3, 5, generator=torch.Generator().manual_seed(9))
        initial = representation_diagnostics(z, z * 2)
        later = representation_diagnostics(z, z * 1.25, initial_delta_rms=initial["initial_delta_rms"])
        self.assertAlmostEqual(later["metrics"]["monitor/delta/rms_vs_initial"], .25, places=5)
        self.assertAlmostEqual(later["metrics"]["monitor/delta/energy_relative"], .0625, places=5)
        torch.testing.assert_close(initial["delta"]["slot_erank"], later["delta"]["slot_erank"])

    def test_fp32_detached_and_input_validation(self):
        z = torch.randn(5, 2, 3, requires_grad=True)
        result = representation_diagnostics(z.double(), z.double() + 1)
        self.assertEqual(result["slot_cka"].dtype, torch.float32)
        self.assertFalse(result["current"]["slot_variance"].requires_grad)
        for bad in (torch.zeros(1, 2, 3), torch.full((3, 2, 3), float("nan"))):
            with self.assertRaises(ValueError):
                representation_diagnostics(bad, bad)
        with self.assertRaises(ValueError):
            representation_diagnostics(z, z, initial_delta_rms=-1)


if __name__ == "__main__":
    unittest.main()

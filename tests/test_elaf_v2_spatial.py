import unittest
import torch

from elaf.v2_model import ELAFV2Config, SpatialAnchoredResampler


class SpatialAnchorsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(239)
        self.config = ELAFV2Config(latent_dim=16, resampler_dim=16, resampler_heads=4)
        self.model = SpatialAnchoredResampler(self.config)

    def test_axis_order_and_local_regions(self):
        # Encode coordinates in values, independently construct expected bin means.
        t = torch.arange(2)[:, None, None]
        y = torch.arange(24)[None, :, None]
        x = torch.arange(24)[None, None, :]
        grid = (10000*t + 100*y + x).float()
        features = grid.reshape(1,1152,1).expand(-1,-1,16)
        actual = self.model.pool_anchors(features)[0,:,0]
        expected = torch.tensor([10000*a + 100*(6*b+2.5) + 6*c+2.5
                                 for a in range(2) for b in range(4) for c in range(4)])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.unique().numel(),32)

    def test_local_impulse_does_not_become_global_pool(self):
        features = torch.zeros(1,1152,16)
        features[:, 24*24+12*24+18] = 36
        pooled = self.model.pool_anchors(features)
        self.assertEqual(pooled[0,:,0].nonzero().flatten().tolist(), [16+2*4+3])
        self.assertEqual(pooled[0,27,0],1)

    def test_exact_residual_and_attention_shape(self):
        features = torch.randn(2,1152,16)
        z, weights = self.model.forward_with_attention(features)
        self.assertEqual(tuple(weights.shape),(2,4,32,1152))
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,4,32))
        with torch.no_grad():
            self.model.output_projection.weight.zero_()
            self.model.output_projection.bias.zero_()
        torch.testing.assert_close(self.model(features),self.model.pool_anchors(features),rtol=0,atol=0)

    def test_fp32_with_autocast_and_trainable_branch(self):
        features = torch.randn(2,1152,16,requires_grad=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            z = self.model(features)
        self.assertEqual(z.dtype,torch.float32)
        z.square().mean().backward()
        self.assertIsNone(features.grad)
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(self.model.query_projection.weight.grad.norm(),0)

    def test_incorrect_layout_rejected(self):
        with self.assertRaises(ValueError): self.model(torch.randn(1,392,16))
        with self.assertRaises(ValueError): ELAFV2Config(grid_shape=(1,24,24))
        with self.assertRaises(ValueError): ELAFV2Config(latent_tokens=16)
        with self.assertRaises(ValueError): ELAFV2Config(resampler_fp32=False)


if __name__ == '__main__': unittest.main()

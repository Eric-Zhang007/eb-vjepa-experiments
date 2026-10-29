"""Information-flow and recomputation tests for the adapted real policy class."""
import copy
import unittest
import torch
from torch import nn
from Roboactualizer.models.roboactualizer.predictive_context import PredictiveContextPolicy


class Backbone(nn.Module):
    embed_dim = 16
    num_patches = 4
    patch_grid = (2, 2)
    def encode_clip_last_group(self, x):
        return x


def model():
    torch.manual_seed(7)
    m = PredictiveContextPolicy(Backbone(), chunk_size=4, future_size=2,
        video_freq_ratio=2, action_dim=3, text_dim=12, proprio_dim=5,
        d_model=24, num_layers=3, num_heads=2, dim_head=12, mlp_dim=48,
        freq_dim=16, predictor_checkpointing=False)
    # Exercise live paths instead of obtaining vacuous zero-head/gate agreement.
    with torch.no_grad():
        nn.init.normal_(m.action_decoder.head.weight, std=.1)
        nn.init.normal_(m.image_decoder.head.weight, std=.1)
        for block in [*m.joint_expert.image_blocks, *m.joint_expert.action_blocks]:
            block.mod[2].fill_(.2); block.mod[5].fill_(.2)
    return m


def batch():
    return dict(z0=torch.randn(2, 4, 16), z_future=torch.randn(2, 2, 4, 16),
        action_chunk=torch.randn(2, 4, 3), proprio=torch.randn(2, 5),
        text_context=torch.randn(2, 6, 12), text_mask=torch.ones(2, 6, dtype=torch.bool),
        action_is_pad=torch.zeros(2, 4, dtype=torch.bool),
        image_is_pad=torch.zeros(2, 3, dtype=torch.bool))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.m = model(); self.b = batch()
        self.noise = torch.randn_like(self.b['action_chunk'])
        self.sigma = torch.full((2, 1, 1), .5)
        self.t = torch.full((2,), 500.)

    def predictions(self, b=None):
        return self.m.prediction_and_targets(b or self.b, self.noise, self.sigma, self.t)

    def test_future_targets_do_not_enter_predictions(self):
        a = self.predictions()
        changed = dict(self.b, z_future=self.b['z_future'] * 100 + 20)
        b = self.predictions(changed)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[3], b[3]))

    def test_future_query_changes_actions_and_receives_action_gradients(self):
        a = self.predictions()[0]
        a.square().mean().backward()
        self.assertGreater(self.m.image_token_encoder.future_query.grad.norm().item(), 1e-9)
        with torch.no_grad():
            self.m.image_token_encoder.future_query.add_(torch.randn_like(self.m.image_token_encoder.future_query))
        b = self.predictions()[0]
        self.assertGreater((a - b).abs().max().item(), 1e-6)

    def test_only_current_kv_is_exported(self):
        c, mask = self.m.image_context_builder(self.b['text_context'], self.b['text_mask'], self.b['proprio'])
        future, kv = self.m._predict_context(self.b['z0'], c, mask)
        self.assertEqual(tuple(future.shape), (2, 2, 4, 16))
        self.assertEqual(len(kv), 3)
        self.assertTrue(all(k.shape == (2, 4, 24) and v.shape == k.shape for k, v in kv))

    def test_checkpointed_gradients_match(self):
        other = copy.deepcopy(self.m); other.predictor_checkpointing = True
        torch.manual_seed(99); a, _ = self.m(self.b); a.backward()
        torch.manual_seed(99); b, _ = other(self.b); b.backward()
        torch.testing.assert_close(a, b)
        for (name, p), (_, q) in zip(self.m.named_parameters(), other.named_parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad, name)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name)

    def test_padding_masks_future_loss(self):
        self.b['image_is_pad'][:, -1] = True
        torch.manual_seed(12); a, _ = self.m(self.b)
        self.b['z_future'][:, -1].fill_(10000)
        torch.manual_seed(12); b, _ = self.m(self.b)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_zero_baseline_uses_identical_mask_reduction(self):
        self.b['image_is_pad'][0, 1:] = True
        self.b['image_is_pad'][1, -1] = True
        with torch.no_grad():
            self.m.image_decoder.head.weight.zero_(); self.m.image_decoder.head.bias.zero_()
        _, logs = self.m(self.b)
        torch.testing.assert_close(logs['image_loss'], logs['world_zero_mse'], rtol=0, atol=0)

    def test_cached_and_uncached_generation_match(self):
        self.m.eval()
        args = (self.b['z0'], self.b['text_context'], self.b['text_mask'], self.b['proprio'])
        torch.manual_seed(22)
        a = self.m.generate(*args, num_inference_steps=4, use_kv_cache=False, use_cuda_graph=False)
        torch.manual_seed(22)
        b = self.m.generate(*args, num_inference_steps=4, use_kv_cache=True, use_cuda_graph=False)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

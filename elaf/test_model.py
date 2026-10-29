"""Run: python -m unittest elaf.test_model -v."""

import io
import os
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from elaf.model import ELAFConfig, ELAFPolicy


class CoupledDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mix = nn.Linear(dim, dim)

    def forward(self, inputs_embeds, attention_mask, **kwargs):
        weights = attention_mask.unsqueeze(-1)
        context = (inputs_embeds * weights).sum(1, keepdim=True) / weights.sum(1, keepdim=True)
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.mix(context))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, image_token_id=31)
        self.embedding = nn.Embedding(32, 16)
        self.model = CoupledDecoder(16)
        self.projector = nn.Linear(4, 16)

    def get_input_embeddings(self):
        return self.embedding

    def get_image_features(self, pixel_values, image_grid_thw):
        return (pixel_values,)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg = ELAFConfig(action_dim=3, action_horizon=2, latent_dim=4, latent_tokens=3, proprio_dim=2)
        self.policy = ELAFPolicy(TinyBackbone(), self.cfg)
        self.batch = dict(input_ids=torch.tensor([[1, 31, 2, 0], [1, 31, 0, 0]]), attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]), pixel_values=torch.randn(2, 4), image_grid_thw=torch.ones(2, 3, dtype=torch.long), proprio=torch.randn(2, 2), actions=torch.randn(2, 2, 3), future_latents=torch.randn(2, 3, 4), action_valid=torch.tensor([[1, 1], [1, 0]]))

    def sample(self, mode="joint"):
        return self.policy.predict_action(self.batch, 4, mode, generator=torch.Generator().manual_seed(11))

    def test_gradients_and_shapes(self):
        self.batch["future_latents"].requires_grad_(True)
        result = self.policy.loss(self.batch)
        result["loss"].backward()
        for layer in (self.policy.action_in[0], self.policy.action_out, self.policy.latent_in[0], self.policy.latent_out, self.policy.backbone.model.mix):
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(self.batch["future_latents"].grad)
        sampled = self.sample()
        self.assertEqual(sampled["actions"].shape, (2, 2, 3))
        self.assertEqual(sampled["future_latents"].shape, (2, 3, 4))

    def test_cross_stream_and_padding(self):
        condition, mask = self.policy._condition(self.batch)
        a, z = torch.randn(2, 2, 3), torch.randn(2, 3, 4, requires_grad=True)
        times = torch.zeros(2)
        va, _ = self.policy.velocity(condition, mask, a, z, times, times)
        gradient = torch.autograd.grad(va.sum(), z)[0]
        self.assertGreater(gradient.abs().sum().item(), 0)
        changed = condition.clone()
        changed[~mask] = 10000
        va2, vz2 = self.policy.velocity(changed, mask, a, z, times, times)
        _, vz = self.policy.velocity(condition, mask, a, z, times, times)
        torch.testing.assert_close(va, va2)
        torch.testing.assert_close(vz, vz2)

    def test_modes_and_serialization(self):
        for mode in ("joint", "fixed_latent_noise", "action_only"):
            result = self.sample(mode)
            torch.testing.assert_close(result["actions"], self.sample(mode)["actions"])
        generator = torch.Generator().manual_seed(11)
        torch.randn((2, 2, 3), generator=generator)
        original_latents = torch.randn((2, 3, 4), generator=generator)
        torch.testing.assert_close(self.sample("fixed_latent_noise")["future_latents"], original_latents)
        self.assertIsNone(self.sample("action_only")["future_latents"])
        buffer = io.BytesIO()
        torch.save(self.policy.state_dict(), buffer)
        buffer.seek(0)
        restored = ELAFPolicy(TinyBackbone(), self.cfg)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        actual = restored.predict_action(self.batch, generator=torch.Generator().manual_seed(11))
        torch.testing.assert_close(actual["actions"], self.sample()["actions"])

    def test_exact_steps_and_endpoint(self):
        calls = []
        def constant_velocity(condition, mask, actions, latents, s, u, action_valid=None):
            calls.append((s.clone(), u.clone()))
            return torch.ones_like(actions), None if latents is None else torch.full_like(latents, 2)
        self.policy.velocity = constant_velocity
        generator = torch.Generator().manual_seed(11)
        a0 = torch.randn((2, 2, 3), generator=generator)
        z0 = torch.randn((2, 3, 4), generator=generator)
        result = self.sample()
        self.assertEqual(len(calls), 4)
        torch.testing.assert_close(result["actions"], a0 + 1)
        torch.testing.assert_close(result["future_latents"], z0 + 2)

    def test_reject_invalid_and_action_only(self):
        self.batch["action_valid"].zero_()
        with self.assertRaises(ValueError):
            self.policy.loss(self.batch)
        with self.assertRaises(ValueError):
            self.policy.predict_action(self.batch, num_steps=0)
        policy = ELAFPolicy(TinyBackbone(), ELAFConfig(action_dim=3, action_horizon=2, proprio_dim=2, action_only=True))
        batch = dict(self.batch)
        batch.pop("future_latents")
        batch.pop("action_valid")
        result = policy.loss(batch)
        self.assertEqual(result["latent_loss"].item(), 0)
        result["loss"].backward()

    @unittest.skipUnless(os.environ.get("ELAF_TEST_NATIVE") == "1", "set ELAF_TEST_NATIVE=1 with Dream-VLX/vla on PYTHONPATH")
    def test_native_dream_decoder(self):
        from dreamvla.configuration_dreamvl import DreamVLAConfig
        from dreamvla.modeling_dreamvl import DreamVLBaseModel
        config = DreamVLAConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, mrope_section=[1, 1, 2], n_action_bins=256, pad_token_id=0)
        config._attn_implementation = "sdpa"
        self.policy.backbone.model = DreamVLBaseModel(config)
        result = self.policy.loss(self.batch)
        result["loss"].backward()
        self.assertGreater(self.policy.latent_in[0].weight.grad.abs().sum().item(), 0)
        sampled = self.sample()
        self.assertTrue(torch.isfinite(sampled["actions"]).all())
        before = sampled["actions"]
        self.batch["input_ids"][self.batch["attention_mask"] == 0] = 11
        torch.testing.assert_close(before, self.sample()["actions"])

    @unittest.skipUnless(os.environ.get("ELAF_TEST_NATIVE") == "1", "native dependencies required")
    def test_native_full_multimodal(self):
        from dreamvla.configuration_dreamvl import DreamVLAConfig
        from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction
        config = DreamVLAConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            mrope_section=[1, 1, 2], n_action_bins=256, norm_stats={},
            pad_token_id=0, image_token_id=31, projector_hidden_act="gelu",
            vision_config=dict(depth=1, embed_dim=16, hidden_size=16, num_heads=2,
                               patch_size=2, spatial_merge_size=2, temporal_patch_size=2))
        config._attn_implementation = "sdpa"
        policy = ELAFPolicy(DreamVLAForActionPrediction(config), self.cfg)
        batch = dict(self.batch)
        batch["pixel_values"] = torch.randn(8, 3 * 2 * 2 * 2)
        batch["image_grid_thw"] = torch.tensor([[1, 2, 2], [1, 2, 2]])
        policy.loss(batch)["loss"].backward()
        self.assertGreater(policy.backbone.visual.patch_embed.proj.weight.grad.abs().sum().item(), 0)
        if torch.cuda.is_available():
            policy = policy.to(device="cuda", dtype=torch.bfloat16)
            batch = {key: value.to(device="cuda", dtype=torch.bfloat16 if value.is_floating_point() else value.dtype) for key, value in batch.items()}
            policy.zero_grad(set_to_none=True)
            result = policy.loss(batch)
            result["loss"].backward()
            self.assertTrue(torch.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()

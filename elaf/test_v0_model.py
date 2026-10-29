"""CPU contract checks: python -m unittest elaf.test_v0_model -v."""

import io
import os
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from elaf.v0_model import ELAFV0Config, ELAFV0Policy


class TinyDecoder(nn.Module):
    def __init__(self, dim, implementation="sdpa"):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.mix = nn.Linear(dim, dim)
        self.last_input = None
        self.last_mask = None

    def forward(self, inputs_embeds, attention_mask, position_ids, **kwargs):
        self.last_input = inputs_embeds.detach().clone()
        self.last_mask = attention_mask.detach().clone()
        self.last_positions = position_ids.detach().clone()
        q = inputs_embeds[:, None]
        mixed = F.scaled_dot_product_attention(q, q, q, attn_mask=attention_mask, is_causal=False)
        return SimpleNamespace(last_hidden_state=inputs_embeds + self.mix(mixed[:, 0]))


class TinyBackbone(nn.Module):
    def __init__(self, implementation="sdpa"):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=24, image_token_id=31)
        self.embedding = nn.Embedding(32, 24)
        self.projector = nn.Linear(4, 24)
        self.model = TinyDecoder(24, implementation)

    def get_input_embeddings(self):
        return self.embedding

    def get_image_features(self, pixel_values, image_grid_thw):
        return (pixel_values,)


class V0ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.cfg = ELAFV0Config(latent_dim=16, resampler_dim=16, resampler_heads=4, proprio_dim=2)
        self.policy = ELAFV0Policy(TinyBackbone(), self.cfg)
        self.batch = {
            "input_ids": torch.tensor([[1, 31, 2, 0], [1, 31, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "pixel_values": torch.randn(2, 4), "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            "proprio": torch.randn(2, 2), "actions": torch.randn(2, 8, 7),
            "current_features": torch.randn(2, 13, 16, requires_grad=True),
            "future_features": torch.randn(2, 13, 16, requires_grad=True),
            "action_valid": torch.tensor([[1] * 8, [1] * 5 + [0] * 3]),
            "future_valid": torch.ones(2, dtype=torch.bool),
        }
        self.noise = dict(delta_noise=torch.randn(2, 32, 16), action_noise=torch.randn(2, 8, 7),
                          u=torch.tensor([0.3, 0.7]), s=torch.tensor([0.6, 0.2]))

    def test_both_losses_reach_shared_resampler_and_backbone(self):
        out = self.policy.loss(self.batch, **self.noise)
        params = list(self.policy.resampler.parameters())
        for name in ("action_loss", "delta_loss"):
            grads = torch.autograd.grad(out[name], params, retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.float().square().sum() for g in grads).sqrt().item(), 0)
        out["loss"].backward()
        for module in (self.policy.current_in, self.policy.delta_in, self.policy.action_in,
                       self.policy.delta_out, self.policy.action_out, self.policy.backbone.model.mix):
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(self.batch["current_features"].grad)
        self.assertIsNone(self.batch["future_features"].grad)

    def test_zero_time_has_no_future_information(self):
        knobs = dict(self.noise, u=torch.zeros(2), s=torch.zeros(2), return_debug=True)
        first = self.policy.loss(self.batch, **knobs)
        inputs = self.policy.backbone.model.last_input.clone()
        changed = dict(self.batch, future_features=torch.randn_like(self.batch["future_features"]) * 5,
                       actions=torch.randn_like(self.batch["actions"]) * 10)
        second = self.policy.loss(changed, **knobs)
        torch.testing.assert_close(inputs, self.policy.backbone.model.last_input, rtol=0, atol=0)
        for key in ("action_velocity", "delta_velocity", "noisy_delta", "noisy_actions"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        torch.testing.assert_close(first["noisy_delta"], self.noise["delta_noise"], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["delta_target"], second["delta_target"]))

    def test_72_tokens_and_bidirectional_visibility(self):
        cond, mask = self.policy._condition(self.batch)
        z = self.policy.resampler(self.batch["current_features"])
        delta = self.noise["delta_noise"].clone().requires_grad_()
        action = self.noise["action_noise"].clone().requires_grad_()
        va, vd = self.policy.velocity(cond, mask, z, delta, action, self.noise["u"], self.noise["s"])
        self.assertEqual(self.policy.extra_tokens, 72)
        self.assertEqual(self.policy.backbone.model.last_input.shape, (2, cond.shape[1] + 72, 24))
        self.assertEqual(va.shape, (2, 8, 7))
        self.assertEqual(vd.shape, (2, 32, 16))
        self.assertGreater(torch.autograd.grad(va.sum(), delta, retain_graph=True)[0].abs().sum().item(), 0)
        # Action tokens occur AFTER delta: this fails with causal attention.
        self.assertGreater(torch.autograd.grad(vd.sum(), action)[0].abs().sum().item(), 0)
        m = self.policy.backbone.model.last_mask
        self.assertTrue(m[:, :, -72:, -72:].all())

    def test_padding_and_missing_future_are_ignored(self):
        batch = dict(self.batch, future_valid=torch.tensor([1, 0]))
        first = self.policy.loss(batch, **self.noise, return_debug=True)
        actions = batch["actions"].clone()
        actions[~batch["action_valid"].bool()] = 10000
        futures = batch["future_features"].detach().clone()
        futures[1] = torch.randn_like(futures[1]) * 1000
        changed = dict(batch, actions=actions, future_features=futures)
        second = self.policy.loss(changed, **self.noise, return_debug=True)
        for key in ("loss", "action_loss", "delta_loss", "action_velocity", "delta_velocity"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        none = self.policy.loss(dict(batch, future_valid=torch.zeros(2)), **self.noise)
        self.assertEqual(none["delta_loss"].item(), 0)
        none["loss"].backward()

    def test_means_and_target_gradient_are_not_detached(self):
        out = self.policy.loss(self.batch, **self.noise, return_debug=True)
        valid = self.batch["action_valid"].bool()[..., None].expand(-1, -1, 7)
        target_a = self.batch["actions"] - self.noise["action_noise"]
        expected_a = (out["action_velocity"] - target_a).square()[valid].mean()
        expected_d = (out["delta_velocity"] - (out["delta_target"] - self.noise["delta_noise"])).square().mean()
        torch.testing.assert_close(out["action_loss"], expected_a)
        torch.testing.assert_close(out["delta_loss"], expected_d)
        grads = torch.autograd.grad(out["delta_loss"], out["future_state"], retain_graph=True)[0]
        self.assertGreater(grads.abs().sum().item(), 0)
        # Force u=0: any gradient to future_state must pass through the target.
        out = self.policy.loss(self.batch, **dict(self.noise, u=torch.zeros(2)), return_debug=True)
        self.assertGreater(torch.autograd.grad(out["delta_loss"], out["future_state"])[0].abs().sum().item(), 0)

    def test_shared_coordinates_and_serialized_scale(self):
        same = dict(self.batch, future_features=self.batch["current_features"])
        out = self.policy.loss(same, **self.noise, return_debug=True)
        torch.testing.assert_close(out["raw_delta"], torch.zeros_like(out["raw_delta"]), rtol=0, atol=0)
        self.policy.delta_scale.fill_(0.125)
        saved = io.BytesIO()
        torch.save(self.policy.state_dict(), saved)
        saved.seek(0)
        other = ELAFV0Policy(TinyBackbone(), self.cfg)
        other.load_state_dict(torch.load(saved, weights_only=True))
        self.assertEqual(other.delta_scale.item(), 0.125)
        expected = self.policy.loss(self.batch, **self.noise)
        actual = other.loss(self.batch, **self.noise)
        torch.testing.assert_close(expected["loss"], actual["loss"], rtol=0, atol=0)

    def test_fp32_resampler_preserves_precision_under_bf16_autocast(self):
        self.cfg.resampler_fp32 = True
        expected_current = self.policy.resampler(self.batch["current_features"])
        expected_future = self.policy.resampler(self.batch["future_features"])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = self.policy.loss(self.batch, **self.noise, return_debug=True)
        for key in ("current_state", "future_state", "raw_delta", "delta_target",
                    "noisy_delta", "delta_velocity_target", "action_velocity_target"):
            self.assertEqual(out[key].dtype, torch.float32, key)
        torch.testing.assert_close(out["current_state"], expected_current, rtol=0, atol=0)
        torch.testing.assert_close(out["raw_delta"], expected_future - expected_current, rtol=0, atol=0)
        torch.testing.assert_close(out["delta_velocity_target"],
                                   out["delta_target"] - self.noise["delta_noise"])
        for key in ("action_loss", "delta_loss"):
            gradients = torch.autograd.grad(out[key], list(self.policy.resampler.parameters()), retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.square().sum() for g in gradients).item(), 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = self.policy.predict_action(self.batch)
        for key in ("current_state", "delta", "future_state"):
            self.assertEqual(prediction[key].dtype, torch.float32, key)

    def test_old_config_retains_autocast_resampler(self):
        self.assertFalse(self.cfg.resampler_fp32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = self.policy.loss(self.batch, **self.noise, return_debug=True)
        self.assertEqual(out["current_state"].dtype, torch.bfloat16)

    def test_four_joint_euler_steps_and_no_inference_future(self):
        calls = []
        def velocity(condition, mask, current, delta, actions, u, s, action_valid=None):
            calls.append((u.clone(), s.clone()))
            return torch.ones_like(actions), torch.full_like(delta, 2)
        self.policy.velocity = velocity
        generator = torch.Generator().manual_seed(9)
        d0 = torch.randn((2, 32, 16), generator=generator)
        a0 = torch.randn((2, 8, 7), generator=generator)
        batch = {k: v for k, v in self.batch.items() if k not in ("future_features", "actions", "future_valid", "action_valid")}
        self.policy.delta_scale.fill_(0.5)
        result = self.policy.predict_action(batch, generator=torch.Generator().manual_seed(9))
        self.assertEqual(len(calls), 4)
        for step, (u, s) in enumerate(calls):
            torch.testing.assert_close(u, s)
            torch.testing.assert_close(u, torch.full((2,), step / 4))
        torch.testing.assert_close(result["actions"], a0 + 1)
        torch.testing.assert_close(result["delta"], d0 + 2)
        torch.testing.assert_close(result["future_state"], result["current_state"] + 0.5 * (d0 + 2))

    def test_eager_padding_mask_and_rejections(self):
        eager = ELAFV0Policy(TinyBackbone("eager"), self.cfg)
        eager.load_state_dict(self.policy.state_dict())
        a = eager.loss(self.batch, **self.noise)
        b = self.policy.loss(self.batch, **self.noise)
        torch.testing.assert_close(a["loss"], b["loss"])
        with self.assertRaises(ValueError):
            self.policy.loss(self.batch, **dict(self.noise, s=torch.ones(2) * 1.1))
        with self.assertRaises(ValueError):
            self.policy.loss(dict(self.batch, action_valid=torch.zeros(2, 8)))
        causal = TinyBackbone()
        causal.model.is_causal = True
        with self.assertRaises(ValueError):
            ELAFV0Policy(causal, self.cfg)

    @unittest.skipUnless(os.environ.get("ELAF_TEST_NATIVE") == "1", "native Dream dependencies required")
    def test_native_dream_multimodal_and_reverse_attention(self):
        from dreamvla.configuration_dreamvl import DreamVLAConfig
        from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction
        config = DreamVLAConfig(vocab_size=32, hidden_size=24, intermediate_size=48,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            mrope_section=[2, 2, 2], n_action_bins=256, norm_stats={},
            pad_token_id=0, image_token_id=31, projector_hidden_act="gelu",
            vision_config=dict(depth=1, embed_dim=24, hidden_size=24, num_heads=2,
                               patch_size=2, spatial_merge_size=2, temporal_patch_size=2))
        config._attn_implementation = "sdpa"
        policy = ELAFV0Policy(DreamVLAForActionPrediction(config), self.cfg)
        batch = dict(self.batch, pixel_values=torch.randn(8, 24),
                     image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 2]]))
        out = policy.loss(batch, **self.noise)
        for key in ("action_loss", "delta_loss"):
            grads = torch.autograd.grad(out[key], list(policy.resampler.parameters()), retain_graph=True)
            self.assertGreater(sum(g.square().sum() for g in grads).item(), 0)
        out["loss"].backward()
        cond, mask = policy._condition(batch)
        z = policy.resampler(batch["current_features"])
        delta = self.noise["delta_noise"].clone().requires_grad_()
        action = self.noise["action_noise"].clone().requires_grad_()
        va, vd = policy.velocity(cond, mask, z, delta, action, self.noise["u"], self.noise["s"])
        self.assertGreater(torch.autograd.grad(va.sum(), delta, retain_graph=True)[0].abs().sum().item(), 0)
        self.assertGreater(torch.autograd.grad(vd.sum(), action)[0].abs().sum().item(), 0)
        # Padding text must not change either generated stream in the actual decoder.
        first = policy.predict_action(batch, generator=torch.Generator().manual_seed(22))
        batch["input_ids"] = batch["input_ids"].clone()
        batch["input_ids"][~batch["attention_mask"].bool()] = 11
        second = policy.predict_action(batch, generator=torch.Generator().manual_seed(22))
        torch.testing.assert_close(first["actions"], second["actions"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

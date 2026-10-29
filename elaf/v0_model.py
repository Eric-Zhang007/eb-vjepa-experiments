"""E-LAF v0: 32 current + 32 delta + 8 action tokens in native Dream.

Checklist: shared resampler; runtime backbone width; bidirectional joint flow;
masked FP32 means; explicit zero-time leakage and separate-gradient assertions.
This module consumes frozen encoder features, never cached resampler outputs.
The caller loads the native Dream backbone and configures its frozen/LoRA weights.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from elaf.model import ELAFPolicy


@dataclass
class ELAFV0Config:
    latent_dim: int = 1024
    latent_tokens: int = 32
    resampler_dim: int = 256
    resampler_heads: int = 8
    action_horizon: int = 8
    action_dim: int = 7
    proprio_dim: int = 8
    action_loss_weight: float = 1.0
    delta_loss_weight: float = 1.0
    delta_scale: float = 1.0
    resampler_fp32: bool = False

    def __post_init__(self):
        for key in ("latent_dim", "latent_tokens", "resampler_dim", "resampler_heads", "action_horizon", "action_dim"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.resampler_dim % self.resampler_heads:
            raise ValueError("resampler_dim must be divisible by resampler_heads")
        if not isinstance(self.proprio_dim, int) or self.proprio_dim < 0:
            raise ValueError("proprio_dim must be a nonnegative integer")
        for key in ("action_loss_weight", "delta_loss_weight", "delta_scale"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")


class SharedResampler(nn.Module):
    """One cross-attention layer, query residual, no self-attention or FFN."""

    def __init__(self, config: ELAFV0Config):
        super().__init__()
        c = config
        self.feature_dim = c.latent_dim
        self.queries = nn.Parameter(torch.randn(c.latent_tokens, c.resampler_dim) * 0.02)
        self.input_norm = nn.LayerNorm(c.latent_dim)
        self.attention = nn.MultiheadAttention(c.resampler_dim, c.resampler_heads,
                                             kdim=c.latent_dim, vdim=c.latent_dim,
                                             dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(c.resampler_dim)
        self.output_projection = nn.Linear(c.resampler_dim, c.latent_dim)

    def forward(self, features):
        if features.ndim != 3 or features.shape[1] < 1 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"encoder features must be [B,N,{self.feature_dim}] with N > 0")
        # Raw frozen-encoder features deliberately terminate the encoder graph.
        features = features.detach().to(device=self.queries.device, dtype=self.queries.dtype)
        if not torch.isfinite(features).all():
            raise ValueError("encoder features contain nonfinite values")
        h = self.input_norm(features)
        q = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        attended, _ = self.attention(q, h, h, need_weights=False)
        return self.output_projection(self.output_norm(attended + q))


class ELAFV0Policy(nn.Module):
    # Reuse only the native visual-language condition construction and validation.
    _condition = ELAFPolicy._condition
    _finite = staticmethod(ELAFPolicy._finite)
    _tensor = ELAFPolicy._tensor

    def __init__(self, backbone, config=None):
        super().__init__()
        self.backbone = backbone
        self.config = config or ELAFV0Config()
        self.hidden_size = int(backbone.config.hidden_size)
        if self.hidden_size < 4 or self.hidden_size % 2:
            raise ValueError("backbone hidden size must be even and at least four")
        c, d = self.config, self.hidden_size
        self.resampler = SharedResampler(c)
        self.current_in = nn.Linear(c.latent_dim, d)
        self.delta_in = nn.Linear(c.latent_dim, d)
        self.action_in = nn.Linear(c.action_dim, d)
        self.delta_out = nn.Linear(d, c.latent_dim)
        self.action_out = nn.Linear(d, c.action_dim)
        self.current_position = nn.Parameter(torch.randn(1, c.latent_tokens, d) * 0.02)
        self.delta_position = nn.Parameter(torch.randn(1, c.latent_tokens, d) * 0.02)
        self.action_position = nn.Parameter(torch.randn(1, c.action_horizon, d) * 0.02)
        self.delta_time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.action_time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        if c.proprio_dim:
            self.proprio_in = nn.Linear(c.proprio_dim, d)
        self.register_buffer("delta_scale", torch.tensor(c.delta_scale, dtype=torch.float32))
        # Native Dream is noncausal, including its Flash Attention implementation.
        # Reject a substituted causal decoder rather than silently training another model.
        if any(getattr(module, "is_causal", False) for module in backbone.model.modules()):
            raise ValueError("E-LAF v0 requires a bidirectional Dream decoder")

    @property
    def extra_tokens(self):
        return 2 * self.config.latent_tokens + self.config.action_horizon

    def _time(self, time):
        half = self.hidden_size // 2
        freq = torch.exp(torch.arange(half, device=time.device, dtype=torch.float32)
                         * (-math.log(10000) / (half - 1)))
        phase = time.float()[:, None] * freq
        return torch.cat((phase.sin(), phase.cos()), -1).to(self.action_position.dtype)

    def _valid(self, value, shape, device, name):
        if value is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if value.shape != shape or not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name} must be binary with shape {shape}")
        return value.to(device=device, dtype=torch.bool)

    def velocity(self, condition, condition_mask, current, delta, actions, u, s, action_valid=None):
        """All three streams attend to each other; time embeddings add no tokens."""
        b, c = condition.shape[0], self.config
        valid = self._valid(action_valid, (b, c.action_horizon), condition.device, "action_valid")
        parts = [condition,
                 self.current_in(current) + self.current_position,
                 self.delta_in(delta) + self.delta_position + self.delta_time(self._time(u))[:, None],
                 self.action_in(actions) + self.action_position + self.action_time(self._time(s))[:, None]]
        embeddings = torch.cat(parts, dim=1)
        # New modules retain FP32 master parameters; native Dream computes in BF16.
        embeddings = embeddings.to(self.backbone.get_input_embeddings().weight.dtype)
        keys = torch.cat((condition_mask.bool(), torch.ones((b, 2 * c.latent_tokens),
                         device=condition.device, dtype=torch.bool), valid), dim=1)
        positions = (keys.long().cumsum(1) - 1).clamp_min(0)
        decoder_config = getattr(self.backbone.model, "config", None)
        implementation = getattr(decoder_config, "_attn_implementation", None)
        decoder_mask = keys
        if implementation in ("sdpa", "eager"):
            allowed = keys[:, None, None, :].expand(-1, 1, embeddings.shape[1], -1)
            decoder_mask = allowed
            if implementation == "eager":
                decoder_mask = torch.zeros_like(allowed, dtype=embeddings.dtype).masked_fill(
                    ~allowed, torch.finfo(embeddings.dtype).min)
        if decoder_config is not None:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
        hidden = self.backbone.model(inputs_embeds=embeddings, attention_mask=decoder_mask,
                                     position_ids=positions, use_cache=False,
                                     return_dict=True).last_hidden_state
        delta_start = condition.shape[1] + c.latent_tokens
        vd = self.delta_out(hidden[:, delta_start:delta_start + c.latent_tokens])
        va = self.action_out(hidden[:, delta_start + c.latent_tokens:])
        return va, vd

    def _draw_time(self, value, b, device, name):
        if value is None:
            return torch.rand(b, device=device)
        value = self._tensor(value, name, (b,), device, torch.float32)
        if not ((value >= 0) & (value <= 1)).all():
            raise ValueError(f"{name} must lie in [0,1]")
        return value

    def _resample(self, features):
        if not self.config.resampler_fp32:
            return self.resampler(features)
        if any(p.dtype != torch.float32 for p in self.resampler.parameters()):
            raise ValueError("resampler_fp32 requires FP32 resampler parameters")
        # Disabling autocast before projection preserves small differences between
        # nearby states; casting already-quantized BF16 outputs cannot recover them.
        with torch.autocast(device_type=self.resampler.queries.device.type, enabled=False):
            return self.resampler(features.float())

    def loss(self, batch, *, u=None, s=None, delta_noise=None, action_noise=None, return_debug=False):
        condition, mask = self._condition(batch)
        b, c = condition.shape[0], self.config
        device, dtype = condition.device, condition.dtype
        zt = self._resample(batch["current_features"])
        zf = self._resample(batch["future_features"])
        if zt.shape[0] != b or zf.shape[0] != b:
            raise ValueError("encoder feature batch size must match conditions")
        action = self._tensor(batch.get("actions"), "actions", (b, c.action_horizon, c.action_dim), device, dtype)
        av = self._valid(batch.get("action_valid"), (b, c.action_horizon), device, "action_valid")
        fv = self._valid(batch.get("future_valid"), (b,), device, "future_valid")
        if not av.any():
            raise ValueError("batch must contain at least one valid action")
        # Invalid future endpoints contribute neither target nor target-conditioned input.
        raw_delta = zf - zt
        delta = (raw_delta.float() / self.delta_scale.float()).to(zt.dtype)
        delta = delta.masked_fill(~fv[:, None, None], 0)
        action = action.masked_fill(~av[:, :, None], 0)
        u = self._draw_time(u, b, device, "u")
        s = self._draw_time(s, b, device, "s")
        en_d = torch.randn_like(delta) if delta_noise is None else self._tensor(delta_noise, "delta_noise", delta.shape, device, delta.dtype)
        en_a = torch.randn_like(action) if action_noise is None else self._tensor(action_noise, "action_noise", action.shape, device, action.dtype)
        du = torch.lerp(en_d, delta, u.to(delta.dtype)[:, None, None])
        at = torch.lerp(en_a, action, s.to(action.dtype)[:, None, None])
        va, vd = self.velocity(condition, mask, zt, du, at, u, s, av)
        # Subtract in FP32 too: target gradients remain connected to the resampler.
        target_a = action.float() - en_a.float()
        target_d = delta.float() - en_d.float()
        la = ((va.float() - target_a).square() * av[:, :, None]).sum() / (av.sum() * c.action_dim)
        ld = ((vd.float() - target_d).square() * fv[:, None, None]).sum() / (fv.sum().clamp_min(1) * c.latent_tokens * c.latent_dim)
        total = c.action_loss_weight * la + c.delta_loss_weight * ld
        self._finite(total, "loss")
        result = {"loss": total, "action_loss": la, "delta_loss": ld}
        if return_debug:
            result.update(current_state=zt, future_state=zf, raw_delta=raw_delta,
                          delta_target=delta, noisy_delta=du, noisy_actions=at,
                          action_velocity=va, delta_velocity=vd,
                          action_velocity_target=target_a, delta_velocity_target=target_d,
                          u=u, s=s)
        return result

    def forward(self, batch, **kwargs):
        return self.loss(batch, **kwargs)

    @torch.no_grad()
    def predict_action(self, batch, num_steps=4, generator=None):
        """Joint Euler integration at u=s; never reads future_features or GT actions."""
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        condition, mask = self._condition(batch)
        current = self._resample(batch["current_features"])
        b, c = condition.shape[0], self.config
        if current.shape[0] != b:
            raise ValueError("encoder feature batch size must match conditions")
        kwargs = dict(device=condition.device, dtype=condition.dtype, generator=generator)
        delta_kwargs = dict(kwargs, dtype=torch.float32) if c.resampler_fp32 else kwargs
        delta = torch.randn((b, c.latent_tokens, c.latent_dim), **delta_kwargs)
        action = torch.randn((b, c.action_horizon, c.action_dim), **kwargs)
        for step in range(num_steps):
            tau = torch.full((b,), step / num_steps, device=condition.device)
            va, vd = self.velocity(condition, mask, current, delta, action, tau, tau)
            action = action + va / num_steps
            delta = delta + vd / num_steps
        future = current + self.delta_scale.to(delta.dtype) * delta
        self._finite(action, "predicted actions")
        self._finite(future, "predicted future state")
        return {"actions": action, "delta": delta, "future_state": future, "current_state": current}

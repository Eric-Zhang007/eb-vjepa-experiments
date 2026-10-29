"""Continuous streams inside Dream-VLA's native bidirectional decoder.

Implementation checklist: explicit contracts; native visual conditioning; coupled
stream loss; exact-step integration; coupling/padding/serialization tests.
The caller owns checkpoint loading, freezing, device and precision decisions.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class ELAFConfig:
    action_dim: int = 7
    action_horizon: int = 8
    latent_dim: int = 1024
    latent_tokens: int = 16
    proprio_dim: int = 8
    lambda_latent: float = 1.0
    latent_zero_probability: float = 0.1
    action_only: bool = False

    def __post_init__(self):
        for name in ("action_dim", "action_horizon", "latent_dim", "latent_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.proprio_dim, int) or self.proprio_dim < 0:
            raise ValueError("proprio_dim must be a nonnegative integer")
        if not math.isfinite(self.lambda_latent) or self.lambda_latent < 0:
            raise ValueError("lambda_latent must be finite and nonnegative")
        if not 0 <= self.latent_zero_probability <= 1:
            raise ValueError("latent_zero_probability must be in [0, 1]")


def _mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))


class ELAFPolicy(nn.Module):
    def __init__(self, backbone, config: ELAFConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config
        dim = backbone.config.hidden_size
        if dim < 4 or dim % 2:
            raise ValueError("backbone hidden size must be even and at least four")
        self.hidden_size = dim
        self.action_in = _mlp(1, dim, dim)
        self.action_out = nn.Linear(dim, 1)
        self.action_time = _mlp(dim, dim, dim)
        self.action_type = nn.Parameter(torch.zeros(1, 1, dim))
        if not config.action_only:
            self.latent_in = _mlp(config.latent_dim, dim, dim)
            self.latent_out = nn.Linear(dim, config.latent_dim)
            self.latent_time = _mlp(dim, dim, dim)
            self.latent_type = nn.Parameter(torch.zeros(1, 1, dim))
        if config.proprio_dim:
            self.proprio_in = _mlp(config.proprio_dim, dim, dim)

    @staticmethod
    def _finite(value, name):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")

    def _tensor(self, value, name, shape, device, dtype):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}")
        self._finite(value, name)
        return value.to(device=device, dtype=dtype)

    def _condition(self, batch):
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        if ids.ndim != 2 or ids.shape[1] == 0 or ids.shape[0] == 0 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be a nonempty integer [B,L] tensor")
        if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("attention_mask must be binary and match input_ids")
        if not mask.bool().any(dim=1).all():
            raise ValueError("each condition must contain a valid token")
        embeds = self.backbone.get_input_embeddings()(ids)
        mask = mask.to(device=embeds.device, dtype=torch.bool)
        image_mask = ids == self.backbone.config.image_token_id
        if image_mask.any() and "pixel_values" not in batch:
            raise ValueError("image placeholders require pixel_values and image_grid_thw")
        if "pixel_values" in batch:
            if "image_grid_thw" not in batch:
                raise ValueError("pixel_values require image_grid_thw")
            features = self.backbone.get_image_features(batch["pixel_values"], batch["image_grid_thw"])
            features = torch.cat(features, dim=0)
            features = self.backbone.projector(features)
            if features.shape != (int(image_mask.sum()), self.hidden_size):
                raise ValueError("image feature count does not match image placeholders")
            embeds = embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(embeds), features.to(embeds.dtype))
        if self.config.proprio_dim:
            proprio = self._tensor(batch.get("proprio"), "proprio", (ids.shape[0], self.config.proprio_dim), embeds.device, embeds.dtype)
            embeds = torch.cat((embeds, self.proprio_in(proprio).unsqueeze(1)), dim=1)
            mask = torch.cat((mask, torch.ones((ids.shape[0], 1), device=mask.device, dtype=torch.bool)), dim=1)
        return embeds, mask

    def _time(self, time):
        half = self.hidden_size // 2
        frequencies = torch.exp(torch.arange(half, device=time.device, dtype=torch.float32) * (-math.log(10000) / (half - 1)))
        phase = time.float().unsqueeze(-1) * frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1).to(self.action_type.dtype)

    def velocity(self, condition, condition_mask, actions, latents, action_time, latent_time, action_valid=None):
        """One shared decoder evaluation; latents=None explicitly removes its stream."""
        b = actions.shape[0]
        cfg = self.config
        action_features = self.action_in(actions.reshape(b, -1, 1)) + self.action_type
        parts = [condition, self.action_time(self._time(action_time)).unsqueeze(1)]
        masks = [condition_mask, torch.ones((b, 1), device=condition.device, dtype=torch.bool)]
        if latents is not None:
            if cfg.action_only:
                raise ValueError("action-only policy has no latent stream")
            parts += [self.latent_time(self._time(latent_time)).unsqueeze(1), self.latent_in(latents) + self.latent_type]
            masks += [torch.ones((b, 1 + cfg.latent_tokens), device=condition.device, dtype=torch.bool)]
        latent_start = sum(p.shape[1] for p in parts) - cfg.latent_tokens if latents is not None else None
        action_start = sum(p.shape[1] for p in parts)
        parts.append(action_features)
        if action_valid is None:
            action_valid = torch.ones((b, cfg.action_horizon), device=condition.device, dtype=torch.bool)
        masks.append(action_valid.repeat_interleave(cfg.action_dim, dim=1))
        attention_mask = torch.cat(masks, dim=1)
        embeddings = torch.cat(parts, dim=1)
        positions = (attention_mask.long().cumsum(1) - 1).clamp_min(0)
        decoder_config = getattr(self.backbone.model, "config", None)
        implementation = getattr(decoder_config, "_attn_implementation", None)
        decoder_mask = attention_mask
        if implementation in ("sdpa", "eager"):
            # Upstream only expands a 2-D SDPA mask on CUDA. Explicit key masks
            # work on CPU too and keep padded queries from all-masked softmax.
            allowed = attention_mask[:, None, None, :].expand(-1, 1, embeddings.shape[1], -1)
            decoder_mask = allowed
            if implementation == "eager":
                decoder_mask = torch.zeros_like(allowed, dtype=embeddings.dtype).masked_fill(~allowed, torch.finfo(embeddings.dtype).min)
        if decoder_config is not None:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
        result = self.backbone.model(inputs_embeds=embeddings, attention_mask=decoder_mask, position_ids=positions, use_cache=False, return_dict=True)
        hidden = result.last_hidden_state
        action_velocity = self.action_out(hidden[:, action_start:]).reshape(b, cfg.action_horizon, cfg.action_dim)
        latent_velocity = None if latents is None else self.latent_out(hidden[:, latent_start:latent_start + cfg.latent_tokens])
        return action_velocity, latent_velocity

    def loss(self, batch):
        condition, mask = self._condition(batch)
        b = condition.shape[0]
        cfg = self.config
        device, dtype = condition.device, condition.dtype
        actions = self._tensor(batch.get("actions"), "actions", (b, cfg.action_horizon, cfg.action_dim), device, dtype)
        valid = batch.get("action_valid", torch.ones((b, cfg.action_horizon), device=device, dtype=torch.bool))
        if valid.shape != (b, cfg.action_horizon) or not ((valid == 0) | (valid == 1)).all() or not valid.bool().any(dim=1).all():
            raise ValueError("action_valid must be binary [B,H], with valid actions in every sample")
        valid = valid.to(device=device, dtype=torch.bool)
        noise = torch.randn_like(actions)
        s = torch.rand(b, device=device)
        noisy_actions = torch.lerp(noise, actions, s.to(dtype)[:, None, None])
        latents = noisy_latents = latent_noise = u = None
        if not cfg.action_only:
            latents = self._tensor(batch.get("future_latents"), "future_latents", (b, cfg.latent_tokens, cfg.latent_dim), device, dtype).detach()
            latent_noise = torch.randn_like(latents)
            u = torch.rand(b, device=device)
            u = u.masked_fill(torch.rand(b, device=device) < cfg.latent_zero_probability, 0)
            noisy_latents = torch.lerp(latent_noise, latents, u.to(dtype)[:, None, None])
        va, vz = self.velocity(condition, mask, noisy_actions, noisy_latents, s, u, valid)
        error = (va.float() - (actions - noise).float()).square()
        action_loss = (error * valid.unsqueeze(-1)).sum() / (valid.sum() * cfg.action_dim)
        latent_loss = action_loss.new_zeros(()) if latents is None else (vz.float() - (latents - latent_noise).float()).square().mean()
        loss = action_loss + cfg.lambda_latent * latent_loss
        self._finite(loss, "loss")
        return {"loss": loss, "action_loss": action_loss, "latent_loss": latent_loss}

    def forward(self, batch):
        return self.loss(batch)

    @torch.no_grad()
    def predict_action(self, batch, num_steps=4, mode="joint", generator=None):
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if mode not in ("joint", "fixed_latent_noise", "action_only"):
            raise ValueError("unknown inference mode")
        if self.config.action_only and mode != "action_only":
            raise ValueError("action-only policy requires mode='action_only'")
        condition, mask = self._condition(batch)
        b, cfg = condition.shape[0], self.config
        kwargs = dict(device=condition.device, dtype=condition.dtype, generator=generator)
        actions = torch.randn((b, cfg.action_horizon, cfg.action_dim), **kwargs)
        latents = None if mode == "action_only" else torch.randn((b, cfg.latent_tokens, cfg.latent_dim), **kwargs)
        for step in range(num_steps):
            s = torch.full((b,), step / num_steps, device=condition.device)
            u = torch.zeros_like(s) if mode == "fixed_latent_noise" else s
            va, vz = self.velocity(condition, mask, actions, latents, s, u)
            actions = actions + va / num_steps
            if mode == "joint":
                latents = latents + vz / num_steps
        self._finite(actions, "predicted actions")
        if latents is not None:
            self._finite(latents, "predicted latent state")
        return {"actions": actions, "future_latents": latents}

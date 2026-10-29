"""Spatial-anchored E-LAF; the original v0/v1 model is unchanged.

Verified cache layout: encode_windows feeds [B,C,4,384,384] without masks;
V-JEPA PatchEmbed3D uses Conv3d(kernel/stride=(2,16,16)), then
flatten(2).transpose(1,2). Thus N=1152 is T,H,W order, W fastest.
Transformer blocks preserve token positions. Never infer this layout from N.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from elaf.v0_model import ELAFV0Config, ELAFV0Policy


@dataclass
class ELAFV2Config(ELAFV0Config):
    grid_shape: tuple = (2, 24, 24)
    residual_alpha: float = 0.1
    resampler_fp32: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.grid_shape = tuple(self.grid_shape)
        if len(self.grid_shape) != 3 or any(
            not isinstance(x, int) or isinstance(x, bool) or x < y
            for x, y in zip(self.grid_shape, (2, 4, 4))
        ):
            raise ValueError("grid_shape must provide at least (2,4,4) verified T,H,W cells")
        if self.latent_tokens != 32:
            raise ValueError("spatial anchors require exactly 2*4*4=32 tokens")
        if not math.isfinite(self.residual_alpha) or self.residual_alpha < 0:
            raise ValueError("residual_alpha must be finite and nonnegative")
        if not self.resampler_fp32:
            raise ValueError("v2 requires FP32 resampler and delta construction")


class SpatialAnchoredResampler(nn.Module):
    """P + alpha W_O CrossAttention(W_Q LN(P)+e, LN(H), LN(H))."""

    def __init__(self, config: ELAFV2Config):
        super().__init__()
        self.feature_dim = config.latent_dim
        self.grid_shape = tuple(config.grid_shape)
        self.residual_alpha = float(config.residual_alpha)
        self.spatial_embedding = nn.Parameter(
            torch.randn(32, config.resampler_dim) * 0.02)
        self.anchor_norm = nn.LayerNorm(self.feature_dim)
        self.query_projection = nn.Linear(self.feature_dim, config.resampler_dim)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.attention = nn.MultiheadAttention(
            config.resampler_dim, config.resampler_heads,
            kdim=self.feature_dim, vdim=self.feature_dim,
            dropout=0.0, batch_first=True)
        self.output_projection = nn.Linear(config.resampler_dim, self.feature_dim)

    @property
    def queries(self):
        # Compatibility with inherited FP32 scope/device selection only.
        return self.spatial_embedding

    def _features(self, features):
        shape = (math.prod(self.grid_shape), self.feature_dim)
        if features.ndim != 3 or tuple(features.shape[1:]) != shape:
            raise ValueError(f"verified T,H,W={self.grid_shape} requires [B,{shape[0]},{shape[1]}]")
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError("SpatialAnchoredResampler parameters must remain FP32")
        features = features.detach().to(device=self.spatial_embedding.device, dtype=torch.float32)
        if not torch.isfinite(features).all():
            raise ValueError("encoder features contain nonfinite values")
        return features

    def _pool(self, features):
        grid = features.reshape(features.shape[0], *self.grid_shape, self.feature_dim)
        pooled = F.adaptive_avg_pool3d(grid.permute(0, 4, 1, 2, 3), (2, 4, 4))
        return pooled.flatten(2).transpose(1, 2)

    def pool_anchors(self, features):
        """Raw spatial residual; slot index is t*16 + row*4 + column."""
        with torch.autocast(device_type=self.spatial_embedding.device.type, enabled=False):
            return self._pool(self._features(features))

    def _forward(self, features, need_weights):
        with torch.autocast(device_type=self.spatial_embedding.device.type, enabled=False):
            features = self._features(features)
            anchors = self._pool(features)
            query = self.query_projection(self.anchor_norm(anchors)) + self.spatial_embedding
            h = self.input_norm(features)
            attended, weights = self.attention(
                query, h, h, need_weights=need_weights, average_attn_weights=False)
            return anchors + self.residual_alpha * self.output_projection(attended), weights

    def forward(self, features):
        return self._forward(features, False)[0]

    def forward_with_attention(self, features):
        return self._forward(features, True)


class ELAFV2Policy(ELAFV0Policy):
    def __init__(self, backbone, config=None):
        config = config or ELAFV2Config()
        super().__init__(backbone, config)
        self.resampler = SpatialAnchoredResampler(config)

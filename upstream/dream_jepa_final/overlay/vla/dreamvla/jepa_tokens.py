"""Dual-view anchored JEPA conditioning; targets stay in frozen encoder space."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def spatial_encoding(dim):
    if dim % 4:
        raise ValueError('2D sin/cos requires hidden size divisible by four')
    row, col = torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')
    freq = torch.exp(-math.log(10000) * torch.arange(dim // 4) / max(dim // 4 - 1, 1))
    return torch.cat([fn(axis.flatten()[:, None] * freq) for axis in (row, col)
                      for fn in (torch.sin, torch.cos)], dim=-1)


class DualViewJEPATokens(nn.Module):
    """Shared FP32 resampler; two independently encoded temporal windows.

    grid_shape must come from encoder metadata, never inferred from token count.
    The backbone-facing order is primary(row-major), wrist(row-major).
    """
    def __init__(self, hidden_size, grid_shape, feature_dim=1024, attention_dim=256):
        super().__init__()
        self.grid_shape = tuple(grid_shape)
        self.feature_dim = feature_dim
        if len(self.grid_shape) != 3 or min(self.grid_shape) < 1:
            raise ValueError('verified T,H,W grid required')
        self.norm = nn.LayerNorm(feature_dim)
        self.query = nn.Linear(feature_dim, attention_dim)
        self.query_position = nn.Parameter(torch.randn(16, attention_dim) * .02)
        self.attention = nn.MultiheadAttention(attention_dim, 8, kdim=feature_dim,
                                               vdim=feature_dim, batch_first=True)
        self.residual_out = nn.Linear(attention_dim, feature_dim)
        self.current_projection = nn.Linear(feature_dim, hidden_size)
        self.future_projection = nn.Sequential(nn.Linear(feature_dim, hidden_size),
                                               nn.GELU(), nn.Linear(hidden_size, hidden_size))
        self.camera = nn.Parameter(torch.randn(2, hidden_size) * .02)
        self.modality = nn.Parameter(torch.randn(2, hidden_size) * .02)
        self.register_buffer('position', spatial_encoding(hidden_size))

    def _validate(self, features):
        if features.ndim != 3 or tuple(features.shape[1:]) != (math.prod(self.grid_shape), self.feature_dim):
            raise ValueError('features do not match verified encoder grid')

    def _pool(self, features):
        self._validate(features)
        grid = features.float().reshape(features.shape[0], *self.grid_shape, self.feature_dim)
        return F.adaptive_avg_pool3d(grid.permute(0, 4, 1, 2, 3), (1, 4, 4)).flatten(2).transpose(1, 2)

    def future_target(self, features):
        with torch.no_grad(), torch.autocast(features.device.type, enabled=False):
            return self._pool(features.detach())

    def _resample(self, features):
        features = features.detach().float()
        anchors = self._pool(features)
        h = self.norm(features)
        q = self.query(self.norm(anchors)) + self.query_position
        attended, _ = self.attention(q, h, h, need_weights=False)
        return anchors + .1 * self.residual_out(attended)

    def forward(self, primary, wrist, noisy_future):
        # Keep this module FP32; only cast its outputs at the backbone boundary.
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('JEPA token module must remain FP32')
        with torch.autocast(primary.device.type, enabled=False):
            views = [self.current_projection(self._resample(x)) + self.position
                     + self.camera[i] + self.modality[0] for i, x in enumerate((primary, wrist))]
            if noisy_future.shape != (primary.shape[0], 16, self.feature_dim):
                raise ValueError('future must have 16 primary-view latent tokens')
            future = self.future_projection(noisy_future.float()) + self.position + self.camera[0] + self.modality[1]
            return torch.cat(views, dim=1), future


def joint_attention_mask(valid, future_start, dropped):
    """SDPA boolean mask. Blocking every nonfuture query prevents indirect leaks."""
    if dropped.shape != (valid.shape[0],):
        raise ValueError('dropout decision must be per sample')
    allowed = valid.bool()[:, None, None, :].expand(-1, 1, valid.shape[1], -1).clone()
    allowed[:, :, :future_start, future_start:] &= ~dropped.bool()[:, None, None, None]
    return allowed

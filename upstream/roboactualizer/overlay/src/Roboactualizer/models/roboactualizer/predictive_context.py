"""V-JEPA-Policy-inspired context K/V interface on the existing action expert.

This is an adaptation, not a reproduction of V-JEPA Policy. Keep RoboActualizer
normalization, RoPE, AdaLN blocks and action flow scheduler. Future learned
queries participate in the image trunk, but actions read only current-position
K/V. Future encoder features are regression targets, never predictor inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .policy import FlowPolicy, attend, rope_apply, build_image_rope_freqs
from Roboactualizer.utils.encoder_identity import checkpoint_sha256, require_encoder_identity


class PredictiveContextPolicy(FlowPolicy):
    def __init__(self, *args, predictor_checkpointing=True, future_loss_type='mse', latent_normalization='channel_stats', **kwargs):
        super().__init__(*args, **kwargs)
        if future_loss_type not in ('l1', 'mse') or latent_normalization not in ('channel_stats', 'token_layernorm', 'target_layernorm'):
            raise ValueError('Unknown future loss or latent normalization')
        self.future_loss_type = future_loss_type
        self.latent_normalization = latent_normalization
        if self.action_only or self.future_size < 1:
            raise ValueError('Predictive context requires future_size > 0 and action_only=False')
        # Shared mask content; temporal/spatial identity is provided by existing 3D RoPE.
        self.image_token_encoder.future_query = nn.Parameter(torch.empty(1, 1, self.d_model))
        nn.init.normal_(self.image_token_encoder.future_query, std=0.02)
        self.predictor_checkpointing = bool(predictor_checkpointing)
        if getattr(self.backbone, 'independent_views', False):
            views = self.backbone.num_cameras
            self.image_token_encoder.view_embedding = nn.Parameter(torch.randn(views, self.d_model) * .02)
            gh, packed_gw = self.backbone.patch_grid
            positions = torch.arange(self.future_size + 1) * self.video_freq_ratio
            freqs = build_image_rope_freqs(positions, gh, packed_gw // views, self.d_model // self.joint_expert.heads)
            self.image_rope_freqs = freqs.reshape(self.future_size + 1, 1, self.num_patches // views, -1).expand(-1, views, -1, -1).reshape((self.future_size + 1) * self.num_patches, 1, -1).contiguous()

    def _normalize_latent(self, z):
        if self.latent_normalization == 'target_layernorm':
            return z.float()
        if self.latent_normalization == 'token_layernorm':
            return F.layer_norm(z.float(), (z.shape[-1],))
        return super()._normalize_latent(z)

    def _predictor_layer(self, block, state, adaln, context, mask, freqs):
        q, k, v, gate, ffn = block.attn_in(state, adaln)
        q = rope_apply(q, freqs, self.joint_expert.heads)
        k = rope_apply(k, freqs, self.joint_expert.heads)
        mixed = attend(q, k, v, self.joint_expert.heads)
        state = block.post_mot(state, mixed, gate, ffn, context, mask)
        # No detach: both objectives can train the predictor. Actions cannot read
        # future-position K/V directly, including at the first transformer layer.
        return state, k[:, :self.num_patches].contiguous(), v[:, :self.num_patches].contiguous()

    def _predict_context(self, z0, image_context, context_mask, return_prediction=True):
        current = self.image_token_encoder.latent_proj(z0)
        future = self.image_token_encoder.future_query.to(current.dtype).expand(
            z0.size(0), self.future_size * self.num_patches, -1)
        state = torch.cat((current, future), dim=1)
        if hasattr(self.image_token_encoder, 'view_embedding'):
            views = self.backbone.num_cameras
            offsets = self.image_token_encoder.view_embedding.repeat_interleave(self.num_patches // views, dim=0)
            state = state + offsets.repeat(self.future_size + 1, 1).to(state.dtype)[None]
        time_emb, adaln = self.image_time_embedder(torch.zeros(z0.size(0), device=z0.device))
        adaln = adaln.unsqueeze(1)
        kv = []
        for block in self.joint_expert.image_blocks:
            # Bind block in the callable: backward recomputation must not use
            # the loop's final layer for all checkpoints.
            def layer(x, a, c, m, f, block=block):
                return self._predictor_layer(block, x, a, c, m, f)
            args = (state, adaln, image_context, context_mask, self.image_rope_freqs)
            if self.training and self.predictor_checkpointing and torch.is_grad_enabled():
                state, k, v = checkpoint(layer, *args, use_reentrant=False)
            else:
                state, k, v = layer(*args)
            kv.append((k, v))
        prediction = None
        if return_prediction:
            prediction = self.image_decoder(state[:, self.num_patches:], time_emb,
                                            self.future_size, self.num_patches)
        return prediction, kv

    def _prefill_z0_cache(self, z0, image_context, context_mask, t_action, mods=None):
        # Called by the existing generate() path, including CUDA-graph prefill.
        return self._predict_context(z0, image_context, context_mask, False)[1]

    def _action_only_forward(self, z0, a_noisy, image_context, action_context,
                             context_mask, t_action, action_context_mask=None):
        kv = self._predict_context(z0, image_context, context_mask, False)[1]
        return self._action_only_forward_cached(
            kv, a_noisy, action_context,
            context_mask if action_context_mask is None else action_context_mask, t_action)

    def prediction_and_targets(self, batch, noise, sigma, timestep):
        """Deterministic diagnostic interface; targets never enter prediction."""
        z0, target = self._encode_window(batch)
        if self.latent_normalization == 'target_layernorm':
            target = F.layer_norm(target.float(), (target.shape[-1],))
        context, mask = self.image_context_builder(
            batch['text_context'], batch['text_mask'], batch['proprio'])
        action_context, _ = self.action_context_builder(
            batch['text_context'], batch['text_mask'], batch['proprio'])
        future, kv = self._predict_context(z0, context, mask)
        noisy = self.action_scheduler.add_noise(batch['action_chunk'], noise, sigma)
        action = self._action_only_forward_cached(kv, noisy, action_context, mask, timestep)
        velocity = self.action_scheduler.x_pred_to_velocity(action, noisy, sigma)
        target_velocity = self.action_scheduler.velocity_target(batch['action_chunk'], noisy, sigma)
        return velocity, target_velocity, future, target.detach()

    def training_loss(self, batch):
        action = batch['action_chunk']
        sigma, timestep = self.action_scheduler.sample_train(action.size(0), action.device)
        velocity, target_velocity, future, target = self.prediction_and_targets(
            batch, torch.randn_like(action), sigma, timestep)
        action_tokens = F.mse_loss(velocity.float(), target_velocity.float(), reduction='none').mean(-1)
        image_error = future.float() - target.float()
        image_tokens = (image_error.abs() if self.future_loss_type == 'l1' else image_error.square()).mean((-1, -2))
        action_loss = self._masked_mean(action_tokens, batch.get('action_is_pad')).mean()
        image_loss = self._masked_mean(image_tokens, batch['image_is_pad'][:, 1:]).mean()
        loss = self.action_weight * action_loss + self.image_predict_weight * image_loss
        with torch.no_grad():
            future_pad = batch['image_is_pad'][:, 1:]
            mse_zero = self._masked_mean(target.float().square().mean((-1, -2)), future_pad).mean()
            current = self._normalize_latent(batch['z0']).float() if 'z0' in batch else None
            if current is not None and self.latent_normalization == 'target_layernorm':
                current = F.layer_norm(current, (current.shape[-1],))
            copy_mse = self._masked_mean(
                (target.float() - current[:, None])
                .square().mean((-1, -2)), future_pad).mean() if 'z0' in batch else loss.new_tensor(float('nan'))
        return loss, dict(action_loss=action_loss.detach(), image_loss=image_loss.detach(),
                          world_zero_mse=mse_zero, world_copy_current_mse=copy_mse,
                          world_zero_l1=self._masked_mean(target.float().abs().mean((-1,-2)),future_pad).mean().detach(),
                          world_prediction_mse=self._masked_mean(image_error.square().mean((-1,-2)),future_pad).mean().detach(),
                          action_sigma_mean=sigma.mean().detach())

    def warm_start_action_only(self, path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        require_encoder_identity(payload.get('metadata', {}).get('backbone', {}),
                                 checkpoint_sha256(self.backbone.ckpt_path), path)
        state = payload['model']
        expected = self._trainable_state_dict()
        missing, unexpected = set(expected) - set(state), set(state) - set(expected)
        allowed = {'image_token_encoder.future_query', 'image_decoder.modulation',
                   'image_decoder.head.weight', 'image_decoder.head.bias'}
        if missing != allowed or unexpected:
            raise ValueError(f'Warm start mismatch: missing={missing}, unexpected={unexpected}')
        for key in state:
            if expected[key].shape != state[key].shape:
                raise ValueError(f'Warm start shape mismatch: {key}')
        merged = self.state_dict(); merged.update(state)
        self.load_state_dict(merged, strict=True)
        return dict(source_step=payload.get('step'), reused_tensors=len(state),
                    new_tensors=sorted(missing))

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        state = payload['model']
        expected = self._trainable_state_dict()
        if set(state) != set(expected):
            raise ValueError('Predictive-context checkpoint keys differ; use explicit warm start for action-only weights')
        require_encoder_identity(payload.get('metadata', {}).get('backbone', {}),
                                 checkpoint_sha256(self.backbone.ckpt_path), path)
        merged = self.state_dict(); merged.update(state)
        self.load_state_dict(merged, strict=True)

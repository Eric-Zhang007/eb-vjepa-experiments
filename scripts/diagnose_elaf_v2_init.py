"""Compare historical random-query and spatial-anchored E-LAF initialization."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from elaf.diagnostics import representation_diagnostics, slot_linear_cka
from elaf.slot_diagnostics import attention_metrics, unique_energy
from elaf.v0_model import ELAFV0Config, SharedResampler
from elaf.v2_diagnostics import diagnose_resampler, v2_model_config
from elaf.v2_model import SpatialAnchoredResampler
from elaf.monitor import monitor_manifest_sha256


def summarize(model, cache, device, batch_size):
    model = model.to(device).float().eval()
    result = diagnose_resampler(model, cache, batch_size=batch_size)
    return result


def spatial_path_check(model, cache, device, batch_size):
    model = model.to(device).float().eval()
    pooled_all, outputs, deltas = [], [], []
    with torch.no_grad():
        for start in range(0, len(cache['sample_ids']), batch_size):
            x = cache['current_features'][start:start+batch_size].to(device).float()
            p = model.pool_anchors(x)
            z, _ = model.forward_with_attention(x)
            pooled_all.append(p.cpu()); outputs.append(z.cpu()); deltas.append((z-p).cpu())
    pooled, z, residual = map(torch.cat, (pooled_all, outputs, deltas))
    cka, valid = slot_linear_cka(pooled)
    off = ~torch.eye(cka.shape[0], dtype=torch.bool)
    vals = cka[off & torch.isfinite(cka)]
    # A synthetic grid labels each temporal/spatial cell with its exact index.
    # Pooling must retain the 2x4x4 location order (t*16 + row*4 + col).
    t, h, w = model.grid_shape
    grid = torch.empty((1, t*h*w, model.feature_dim), dtype=torch.float32, device=device)
    for ti in range(t):
        for yi in range(h):
            for xi in range(w):
                index = ti*16 + (yi*4//h)*4 + (xi*4//w)
                grid[0, (ti*h+yi)*w+xi, :] = float(index)
    pooled_grid = model.pool_anchors(grid)[0, :, 0]
    expected = torch.arange(32, dtype=pooled_grid.dtype, device=device)
    grid_order_ok = bool(torch.allclose(pooled_grid, expected, atol=0, rtol=0))
    return {
        'pool_anchor_shape': list(pooled.shape),
        'pool_anchor_unique_slot_energy': unique_energy(pooled),
        'pool_anchor_slot_variance_mean': pooled.float().var(0, unbiased=False).mean(-1).mean().item(),
        'pool_anchor_slot_cka_offdiag_mean': vals.mean().item(),
        'pool_anchor_cka_valid_slots': int(valid.sum().item()),
        'output_shape': list(z.shape),
        'skip_residual_rms': residual.square().mean().sqrt().item(),
        'skip_over_anchor_rms': (residual.square().mean().sqrt()/pooled.square().mean().sqrt().clamp_min(1e-30)).item(),
        'output_minus_pool_rms': (z-pooled).square().mean().sqrt().item(),
        'synthetic_temporal_row_column_order_exact': grid_order_ok,
        'synthetic_pooled_slot_ids': pooled_grid.cpu().tolist(),
        'note': 'output = pool_anchors + alpha * output_projection(cross_attention); anchor path is unscaled and preserved',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='E-LAF v2 JSON config')
    parser.add_argument('--old-snapshot', type=Path, required=True, help='v1 random-query resampler step-zero snapshot')
    parser.add_argument('--new-snapshot', type=Path, help='exact v2 step-zero resampler snapshot; recommended')
    parser.add_argument('--monitor', type=Path, required=True, help='same fixed 1024-pair cache used by prior run')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    config = config.get('config', config)
    cache = torch.load(args.monitor, map_location='cpu', weights_only=True)
    cache_sha = monitor_manifest_sha256(cache)
    old_payload = torch.load(args.old_snapshot, map_location='cpu', weights_only=True)
    if old_payload.get('monitor_manifest_sha256') != cache_sha:
        raise ValueError('Historical snapshot was created against a different fixed monitor cache')

    old_cfg = ELAFV0Config(**old_payload['config'])
    old = SharedResampler(old_cfg).to(args.device).float().eval()
    old.load_state_dict(old_payload['state_dict'], strict=True)
    old_result = summarize(old, cache, args.device, args.batch_size)

    spatial_cfg = v2_model_config(config)
    spatial = SpatialAnchoredResampler(spatial_cfg).to(args.device).float().eval()
    if args.new_snapshot:
        new_payload = torch.load(args.new_snapshot, map_location='cpu', weights_only=True)
        if new_payload.get('monitor_manifest_sha256') != cache_sha:
            raise ValueError('V2 snapshot was created against a different fixed monitor cache')
        if int(new_payload.get('training_step', -1)) != 0:
            raise ValueError('New snapshot must be step zero')
        spatial.load_state_dict(new_payload['state_dict'], strict=True)
    else:
        torch.manual_seed(int(config.get('seed', 239)))
        spatial = SpatialAnchoredResampler(spatial_cfg).to(args.device).float().eval()
    spatial_result = summarize(spatial, cache, args.device, args.batch_size)
    path = spatial_path_check(spatial, cache, args.device, args.batch_size)

    output = {
        'schema_version': 1,
        'protocol': 'same ordered fixed monitor cache; same frozen JEPA features; FP32; eval; step-zero untrained modules',
        'monitor_samples': len(cache['sample_ids']),
        'monitor_manifest': cache.get('manifest'),
        'old_checkpoint': str(args.old_snapshot),
        'old_training_step': old_payload.get('training_step'),
        'v2_config': {'grid_shape': list(spatial_cfg.grid_shape), 'residual_alpha': spatial_cfg.residual_alpha,
                      'resampler_dim': spatial_cfg.resampler_dim, 'resampler_heads': spatial_cfg.resampler_heads},
        'random_query_init': {'metrics': old_result['metrics'], 'initial_delta_rms': old_result['initial_delta_rms']},
        'new_checkpoint': str(args.new_snapshot) if args.new_snapshot else None,
        'spatial_anchored_init': {'metrics': spatial_result['metrics'], 'initial_delta_rms': spatial_result['initial_delta_rms'],
                                  'spatial_path': path},
        'definitions': {'unique_slot_energy': 'same sample-centered metric as prior E-LAF slot diagnostics',
                        'query_jsd': 'pairwise query JSD in nats; also reported per-head, matching prior diagnostics',
                        'effective_rank': 'entropy of covariance eigenvalues; structural rank cap disabled for anchored skip output'},
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({'output': str(args.output), 'random_unique_current': old_result['metrics']['monitor/z/unique_slot_energy'],
                      'random_unique_delta': old_result['metrics']['monitor/delta/unique_slot_energy'],
                      'spatial_unique_current': spatial_result['metrics']['monitor/z/unique_slot_energy'],
                      'spatial_unique_delta': spatial_result['metrics']['monitor/delta/unique_slot_energy'],
                      'spatial_path': path}, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

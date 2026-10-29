"""Full native-model GPU preflight: three updates vs two plus complete resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

from elaf.v0_training import atomic_json, sha256


def compare(left, right, path='state'):
    maximum = 0.
    if torch.is_tensor(left):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise AssertionError(f'{path}: shape/dtype mismatch')
        if left.is_floating_point():
            maximum = (left.float() - right.float()).abs().max().item() if left.numel() else 0.
            if not torch.allclose(left, right, atol=2e-5, rtol=2e-4):
                raise AssertionError(f'{path}: resumed tensor differs, max_abs={maximum}')
        elif not torch.equal(left, right):
            raise AssertionError(f'{path}: integer tensor mismatch')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{path}: keys differ')
        for key in left:
            maximum = max(maximum, compare(left[key], right[key], f'{path}.{key}'))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: lengths differ')
        for i, (a, b) in enumerate(zip(left, right)):
            maximum = max(maximum, compare(a, b, f'{path}[{i}]'))
    elif left != right:
        raise AssertionError(f'{path}: values differ')
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--nproc', type=int, default=2)
    parser.add_argument('--plan', type=Path, help='Watcher plan whose complete source identity must be verified')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    plan = json.loads(args.plan.read_text()) if args.plan else None
    def verify_plan_sources():
        if plan is not None:
            for path, expected in plan['source_sha256'].items():
                if sha256(path) != expected:
                    raise ValueError(f'Watcher source changed: {path}')
    verify_plan_sources()
    # Preserve actual accumulation, precision, schedule and batch; truncate via CLI only.
    base = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc-per-node={args.nproc}', '-m', 'elaf.v0_train', '--config', str(args.config), '--wandb-mode', 'disabled']
    uninterrupted = args.output / 'uninterrupted'
    resumed = args.output / 'resumed'
    for name, command in [
        ('uninterrupted', base + ['--output', str(uninterrupted), '--stop-after-updates', '3']),
        ('interrupted', base + ['--output', str(resumed), '--stop-after-updates', '2']),
        ('resumed', base + ['--output', str(resumed), '--resume', str(resumed / 'checkpoints/latest'), '--stop-after-updates', '3'])]:
        with (args.output / f'{name}.log').open('w') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    left = torch.load(uninterrupted / 'checkpoints/latest/training.pt', map_location='cpu', weights_only=False)
    right = torch.load(resumed / 'checkpoints/latest/training.pt', map_location='cpu', weights_only=False)
    difference = compare(left, right)
    metadata = json.loads((uninterrupted / 'checkpoints/latest/progress.json').read_text())
    other = json.loads((resumed / 'checkpoints/latest/progress.json').read_text())
    for key in ('step', 'micro_step', 'window_count'):
        if metadata['progress'][key] != other['progress'][key]:
            raise AssertionError(f'Progress differs: {key}')
    for rank in range(args.nproc):
        a = torch.load(uninterrupted / f'checkpoints/latest/rank_{rank}.pt', map_location='cpu', weights_only=False)
        b = torch.load(resumed / f'checkpoints/latest/rank_{rank}.pt', map_location='cpu', weights_only=False)
        compare(a['rng']['torch'], b['rng']['torch'])
        compare(a['rng']['cuda'], b['rng']['cuda'])
        if a['micro_step'] != b['micro_step']:
            raise AssertionError('Rank cursor differs')
    verify_plan_sources()
    atomic_json(args.receipt, dict(status='passed', config_sha256=sha256(args.config),
        source_sha256=plan['source_sha256'] if plan is not None else metadata['contract']['source_sha256'],
        cache_manifest_sha256=sha256(config['manifest']), full_native_model=True,
        world_size=args.nproc, updates=3, resume_update=2,
        max_absolute_state_difference=difference, tensor_atol=2e-5, tensor_rtol=2e-4,
        evidence_directory=str(args.output)))


if __name__ == '__main__':
    main()

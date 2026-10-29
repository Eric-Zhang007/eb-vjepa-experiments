"""Compare complete-episode throughput with identical policy, tasks, and protocol."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-config', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--benchmark', choices=['clean','plus','pro'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--execute-steps', type=int, required=True)
    parser.add_argument('--sizes', default='1,4,8,16,32')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(',')]
    episodes = json.loads(Path(args.manifest).read_text())
    if min(sizes) < 1 or max(sizes) > len(episodes):
        parser.error('Use at least as many episodes as the largest concurrency')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for size in sizes:
        out = args.output / f'env_{size}'
        subprocess.run([sys.executable, '-m', 'elaf.evaluation.launch',
            '--policy-config', args.policy_config, '--manifest', args.manifest,
            '--benchmark', args.benchmark, '--gpus', str(args.gpu), '--execute-steps', str(args.execute_steps),
            '--environments', str(size), '--batch-size', str(size), '--output', str(out)], check=True)
        result = json.loads((out/'summary.json').read_text())
        group = result['groups'][0]
        results.append(dict(environments=size, batch_limit=size,
            episodes_per_hour_including_load=result['episodes_per_hour'],
            episodes_per_hour=group['episodes_per_hour'],
            control_steps_per_second=group['control_steps_per_second'],
            mean_batch=group['mean_inference_batch'], max_batch=group['max_inference_batch'],
            episodes=result['episodes'], successes=result['successes'],
            complete=True))
        (args.output/'comparison.json').write_text(json.dumps(results, indent=2))
    best = max(results, key=lambda x:x['episodes_per_hour'])
    (args.output/'recommendation.json').write_text(json.dumps({
        'best_observed':best, 'scope':'this policy, manifest and hardware only; repeat on full E-LAF model',
        'protocol_unchanged':True}, indent=2))


if __name__ == '__main__':
    main()

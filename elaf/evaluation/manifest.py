"""Create an explicit evaluation episode list without launching simulation."""
import argparse
import json
from pathlib import Path
from elaf.evaluation.libero_env import make_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['clean', 'plus', 'pro'], required=True)
    parser.add_argument('--source', type=Path, help='Existing Plus/Pro selection manifest')
    parser.add_argument('--clean-trials', type=int, default=5)
    parser.add_argument('--limit', type=int, help='Explicit smoke/throughput subset size')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.clean_trials < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Positive trials/limit required')
    rows = make_manifest(args.benchmark, args.source)
    if args.benchmark == 'clean' and args.clean_trials != 5:
        tasks = [r for r in rows if r['trial'] == 0]
        rows = [dict(task, trial=trial, seed=239 + i*args.clean_trials + trial,
                     env_seed=239+i*args.clean_trials)
                for i, task in enumerate(tasks) for trial in range(args.clean_trials)]
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(rows, stream, indent=2)
    print(f'{len(rows)} episodes written to {args.output}')


if __name__ == '__main__':
    main()

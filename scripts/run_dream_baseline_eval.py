"""Run the completed Dream baseline through the existing parallel evaluator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--environments', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--benchmarks', nargs='+', choices=['clean', 'plus'], default=['clean', 'plus'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / 'status.json'

    def status(**values):
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, indent=2))
        temporary.replace(status_path)

    cfg = json.loads((args.root / 'elaf/evaluation/configs/dream_vla_official_mixed.json').read_text())
    cfg.update(checkpoint=str(args.checkpoint), max_batch_size=args.batch_size, batch_wait_ms=0)
    policy = args.output / 'policy.json'
    policy.write_text(json.dumps(cfg, indent=2))
    metadata = json.loads((args.checkpoint / 'checkpoint_metadata.json').read_text())
    if metadata['optimizer_update'] != 200000:
        raise ValueError('Expected the completed 200000-update baseline')
    import wandb
    run = wandb.init(mode='disabled', project='eb_vjepa', entity='zjc010100000-xi-an-jiaotong-university-',
        name='dream-baseline-200k-libero200-plus280', job_type='evaluation',
        group='dream-baseline-200k', dir=str(args.output),
        config=dict(policy=cfg, gpus=[6, 7], environments_per_gpu=args.environments,
                    batch_size=args.batch_size, execute_steps=8,
                    protocol='exploratory clean200 and Plus280; official task.language; Dream simulator environment; settle10; clean official horizons; flow4', checkpoint_metadata=metadata))
    (args.output / 'run_identity.json').write_text(json.dumps(dict(wandb_run_id=run.id, url=run.url), indent=2))
    environment = dict(os.environ, PYTHONPATH=str(args.root), OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    started = time.monotonic()
    summaries = {}
    try:
        for benchmark in args.benchmarks:
            count = {'clean': 200, 'plus': 280}[benchmark]
            manifest = args.root / 'experiments/elaf/evaluation_manifests' / (benchmark + '.json')
            specs = json.loads(manifest.read_text())
            if len(specs) != count:
                raise ValueError(f'Unexpected {benchmark} selection size: {len(specs)}')
            argv = [sys.executable, '-m', 'elaf.evaluation.launch', '--root', str(args.root),
                '--policy-config', str(policy), '--manifest', str(manifest), '--benchmark', benchmark,
                '--gpus', '6,7', '--environments', str(args.environments), '--batch-size', str(args.batch_size),
                '--batch-wait-ms', '5', '--execute-steps', '8', '--output', str(args.output / benchmark)]
            (args.output / (benchmark + '_command.json')).write_text(json.dumps(argv, indent=2))
            with (args.output / (benchmark + '.log')).open('w') as stream:
                child = subprocess.Popen(argv, cwd=args.root, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                status(state='running', benchmark=benchmark, child_pid=child.pid, wandb_run_id=run.id)
                if child.wait() != 0:
                    raise RuntimeError(f'{benchmark} evaluator failed; inspect {benchmark}.log and GPU logs')
            summary = json.loads((args.output / benchmark / 'summary.json').read_text())
            summaries[benchmark] = summary
            run.log({f'eval/{benchmark}/{key}': value for key, value in summary.items()
                     if isinstance(value, (int, float)) and not isinstance(value, bool)})
            run.summary[f'eval/{benchmark}/success_rate'] = summary['successes'] / summary['episodes']
        (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
        status(state='completed', seconds=time.monotonic() - started, wandb_run_id=run.id)
    except BaseException as error:
        status(state='failed', error=str(error), seconds=time.monotonic() - started, wandb_run_id=run.id)
        run.finish(exit_code=1)
        raise
    run.finish()


if __name__ == '__main__':
    main()

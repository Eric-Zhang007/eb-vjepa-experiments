"""Consume atomic resampler snapshots on CPU in a separate W&B run.

Checklist: fixed cache and parent identity; step-zero first; exclusive worker;
durable outputs and upload markers; stop after trainer exits and queue drains.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from elaf.monitor import discover_snapshots, run_monitor_snapshot, log_to_wandb, _atomic_write
from elaf.v0_model import ELAFV0Config, SharedResampler


def pending_snapshots(directory, output):
    return [p for p in discover_snapshots(directory)
            if not (Path(output) / (p.stem + '.uploaded')).exists()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--project', default='eb_vjepa')
    parser.add_argument('--entity', default='zjc010100000-xi-an-jiaotong-university-')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    if args.threads < 1 or args.poll_seconds <= 0:
        parser.error('positive threads and polling interval required')
    import fcntl
    import wandb
    args.run_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.run_dir / 'monitor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    identity_path = args.run_dir / 'run_identity.json'
    while not identity_path.exists():
        time.sleep(args.poll_seconds)
    identity = json.loads(identity_path.read_text())
    parent = identity['parent_run_id']
    cache = torch.load(identity['monitor_path'], map_location='cpu', weights_only=True)
    output = args.run_dir / 'diagnostics'
    output.mkdir(exist_ok=True)
    run_record = output / 'wandb_run.json'
    previous = json.loads(run_record.read_text()) if run_record.exists() else None
    if previous and previous['parent_run_id'] != parent:
        raise ValueError('Diagnostics parent run changed')
    run = wandb.init(project=args.project, entity=args.entity,
                     name=f'elaf-v0-diagnostics-{parent}', group=f'elaf-v0-{parent}',
                     job_type='representation-diagnostics',
                     id=None if previous is None else previous['id'],
                     resume=None if previous is None else 'must',
                     config={'parent_run_id': parent,
                             'monitor_manifest_sha256': identity['monitor_manifest_sha256']})
    _atomic_write(run_record, lambda f: f.write(json.dumps({'id': run.id, 'parent_run_id': parent}).encode()))
    try:
        while True:
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('Training identity changed during monitoring')
            for snapshot in pending_snapshots(identity['snapshot_dir'], output):
                payload = torch.load(snapshot, map_location='cpu', weights_only=True)
                config = ELAFV0Config(**payload['config'])
                step = payload['training_step']
                report = run_monitor_snapshot(SharedResampler(config), cache, snapshot, output,
                                              representation_rank_cap=config.resampler_dim)
                result = torch.load(output / f'diagnostics-step-{step:09d}.pt',
                                    map_location='cpu', weights_only=True)
                log_to_wandb(run, result, training_step=step, parent_run_id=parent)
                _atomic_write(output / (snapshot.stem + '.uploaded'),
                              lambda f: f.write(json.dumps({'training_step': step,
                                  'snapshot_sha256': report['snapshot_sha256'], 'run_id': run.id}).encode()))
                print(json.dumps({'training_step': step, 'status': 'logged'}), flush=True)
            status_path = args.run_dir / 'status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('status') in ('completed', 'paused', 'failed'):
                if not pending_snapshots(identity['snapshot_dir'], output):
                    break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == '__main__':
    main()

"""Run E-LAF v2 as a resumable three-stage job with a 15k decision gate.

The trainer is deliberately restarted at stage boundaries. This keeps DDP,
``requires_grad`` and optimizer parameter groups mathematically explicit while
preserving the checkpointed data cursor, RNG and W&B run id.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BOUNDARIES = {3000, 10000}
DIAGNOSTIC_STEPS = {0, 1000, 2000, 3000, 5000, 10000, 11000, 12000, 15000}


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def wait_for(path: Path, timeout: float, description: str, process=None):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'process exited with code {process.returncode} while waiting for {description}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'timed out waiting for {description}: {path}')
        time.sleep(5)


def launch_trainer(args, stage, resume=None):
    log = args.output / f'trainer-stage{stage}.log'
    command = [args.python, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node', str(args.nproc), '-m', 'elaf.v2_train',
               '--config', str(args.config), '--output', str(args.output),
               '--wandb-mode', args.wandb_mode]
    if resume is not None:
        command.extend(['--resume', str(resume)])
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = args.trainer_gpus
    environment.setdefault('OMP_NUM_THREADS', '4')
    environment.setdefault('TOKENIZERS_PARALLELISM', 'false')
    stream = log.open('a', encoding='utf-8')
    process = subprocess.Popen(command, cwd=args.repo, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
    return process, stream


def launch_diagnostics(args):
    log = args.output / 'diagnostics-worker.log'
    command = [args.python, '-m', 'elaf.v2_diagnostics', '--run-dir',
               str(args.output), '--watch', '--device', 'cuda:0']
    environment = dict(os.environ)
    environment['CUDA_VISIBLE_DEVICES'] = args.diagnostic_gpu
    environment.setdefault('OMP_NUM_THREADS', '4')
    environment.setdefault('MKL_NUM_THREADS', '4')
    stream = log.open('a', encoding='utf-8')
    process = subprocess.Popen(command, cwd=args.repo, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True)
    return process, stream


def wait_diagnostic(args, step, process):
    if step not in DIAGNOSTIC_STEPS:
        return
    report = args.output / 'v2_diagnostics' / f'report-{step:09d}.md'
    receipt = args.output / 'v2_diagnostics' / f'step-{step:06d}.done.json'
    deadline = time.monotonic() + args.diagnostic_timeout
    while not (report.exists() and receipt.exists()):
        if process.poll() is not None:
            raise RuntimeError(f'diagnostics worker exited with code {process.returncode} before step {step} report')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'timed out waiting for diagnostics at step {step}')
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--nproc', type=int, default=6)
    parser.add_argument('--trainer-gpus', default='0,1,2,3,4,5')
    parser.add_argument('--diagnostic-gpu', default='6')
    parser.add_argument('--old-snapshot', type=Path, required=True)
    parser.add_argument('--monitor', type=Path, required=True)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    parser.add_argument('--diagnostic-timeout', type=float, default=6 * 3600)
    parser.add_argument('--snapshot-timeout', type=float, default=2 * 3600)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    args.old_snapshot = args.old_snapshot.resolve()
    args.monitor = args.monitor.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    supervisor_log = args.output / 'supervisor.log'

    def log(message):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}'
        print(line, flush=True)
        with supervisor_log.open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    existing = [p for p in args.output.iterdir() if p.name not in {'supervisor.log'}]
    if existing:
        raise FileExistsError(f'fresh v2 output is not empty: {args.output}')
    if not args.old_snapshot.is_file() or not args.monitor.is_file():
        raise FileNotFoundError('historical snapshot and monitor cache are required')

    trainer = None
    trainer_stream = None
    diagnostics = None
    diagnostic_stream = None
    try:
        trainer, trainer_stream = launch_trainer(args, 1)
        log(f'launched Stage 1 trainer pid={trainer.pid}')
        wait_for(args.output / 'resampler_snapshots' / 'resampler-step-000000000.pt',
                 args.snapshot_timeout, 'exact v2 step-0 resampler snapshot', trainer)
        log('step-0 snapshot published; running spatial initialization comparison on diagnostic GPU')
        init_output = args.output / 'ELAF_V2_INIT_DIAGNOSTICS.json'
        init_log = (args.output / 'initialization-diagnostic.log').open('a', encoding='utf-8')
        init_env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.diagnostic_gpu,
                        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        init_cmd = [args.python, 'scripts/diagnose_elaf_v2_init.py', '--config',
                    str(args.config), '--old-snapshot', str(args.old_snapshot),
                    '--new-snapshot', str(args.output / 'resampler_snapshots' /
                                         'resampler-step-000000000.pt'),
                    '--monitor', str(args.monitor), '--output', str(init_output),
                    '--device', 'cuda:0']
        result = subprocess.run(init_cmd, cwd=args.repo, env=init_env,
                                stdout=init_log, stderr=subprocess.STDOUT,
                                timeout=args.diagnostic_timeout)
        init_log.close()
        if result.returncode:
            raise RuntimeError(f'initialization diagnostic failed: {result.returncode}')
        log(f'initialization comparison saved to {init_output}')
        diagnostics, diagnostic_stream = launch_diagnostics(args)
        log(f'launched diagnostics worker pid={diagnostics.pid} on GPU {args.diagnostic_gpu}')

        stage = 1
        while True:
            return_code = trainer.wait()
            trainer_stream.close()
            status = read_json(args.output / 'status.json', {}) or {}
            current_step = int(status.get('step', -1))
            state = status.get('status')
            log(f'trainer Stage {stage} exited code={return_code}, status={state}, step={current_step}')
            if return_code != 0 or state == 'failed':
                raise RuntimeError(f'trainer failed at step {current_step}; inspect trainer-stage{stage}.log')
            if state in {'stage_boundary', 'decision_required'}:
                wait_diagnostic(args, current_step, diagnostics)
            if state != 'stage_boundary' or current_step not in BOUNDARIES:
                log(f'run stopped at step {current_step} with status={state}; no automatic continuation')
                break
            stage += 1
            checkpoint = args.output / 'checkpoints' / 'latest'
            wait_for(checkpoint / 'training.pt', args.snapshot_timeout,
                     f'latest checkpoint at stage boundary {current_step}')
            trainer, trainer_stream = launch_trainer(args, stage, checkpoint)
            log(f'resumed Stage {stage} from step {current_step}, pid={trainer.pid}')
    finally:
        if trainer_stream is not None and not trainer_stream.closed:
            trainer_stream.close()
        if diagnostic_stream is not None and not diagnostic_stream.closed:
            diagnostic_stream.close()
        if diagnostics is not None and diagnostics.poll() is None:
            # Let the worker finish queued diagnostics at a decision gate. For
            # a hard trainer failure, terminate it so the supervisor can exit.
            status = read_json(args.output / 'status.json', {}) or {}
            if status.get('status') in {'failed'}:
                diagnostics.terminate()
    log('supervisor finished')


if __name__ == '__main__':
    main()

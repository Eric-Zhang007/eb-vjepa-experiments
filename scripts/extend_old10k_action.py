"""Single-owner full-state continuation of the completed old10k action head."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    old = root / 'experiments/action_only/ctp_full_010000_handoff'
    out = root / 'experiments/action_only/old10k_extend40k'
    out.mkdir(exist_ok=True)
    if args.detach:
        with (out / 'supervisor.log').open('a') as log:
            p = subprocess.Popen([sys.executable, __file__], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({'pid': p.pid, 'output': str(out)}))
        return
    lock = (out / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'status.json').exists():
        raise RuntimeError('Existing continuation; inspect instead of duplicating')
    state = dict(pid=os.getpid(), started_unix=time.time())
    def status(**kw):
        state.update(kw, updated_unix=time.time())
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out / 'status.json')
    child = None
    workers = []
    try:
        source = old / 'train_action_only_010000/checkpoints/state/latest'
        saved = json.loads((source / 'trainer_state.json').read_text())
        if saved['global_step'] != 21360 or saved['resume_contract']['world_size'] != 8:
            raise RuntimeError('Unexpected source state')
        shutil.copy2(source / 'trainer_state.json', out / 'source_trainer_state.json')
        train = out / 'train'
        original = json.loads((old / 'formal_train_status.json').read_text())['command']
        command = [x for x in original if not x.startswith(('output_dir=', 'hydra.run.dir='))]
        command += [f'resume={source}', 'max_steps=40000', 'save_every=1000',
                    f'output_dir={train}', f'hydra.run.dir={train}/hydra',
                    '+continuation_schedule.start_step=21360', '+continuation_schedule.warmup_steps=200',
                    '+continuation_schedule.initial_lr=2e-6', '+continuation_schedule.peak_lr=2e-5',
                    '+continuation_schedule.final_lr=2e-6']
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'OMP_NUM_THREADS': '4',
               'TMPDIR': str(root / 'cache/tmp'), 'WANDB_DIR': str(root / 'wandb'), 'WANDB_MODE': 'online',
               'HF_HOME': str(root / 'cache/huggingface'),
               'PYTHONPATH': f'{root}/repos/RoboActualizer/src:{root}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages'}
        status(phase='waiting_for_cache', command=command, source_step=21360, target_step=40000)
        while True:
            cache = json.loads((old / 'cache_retry1/full_cache_status.json').read_text())
            if cache['phase'] == 'complete':
                break
            if cache['phase'] == 'failed' or not Path(f"/proc/{cache['pid']}").exists():
                raise RuntimeError('Cache failed or owner lost')
            time.sleep(15)
        cache_dir = Path(cache['cache'])
        if len(list(cache_dir.glob('*/episode_*.pt'))) != 1692:
            raise RuntimeError('Incomplete episode cache')
        stats_hash = hashlib.sha256((cache_dir / 'latent_norm_stats.json').read_bytes()).hexdigest()
        if stats_hash != saved['resume_contract']['latent_stats_sha256']:
            raise RuntimeError('Original normalization changed')
        while True:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            if all(int(x) < 1000 for x in memory.splitlines()):
                break
            status(phase='waiting_for_idle_gpus')
            time.sleep(15)
        with (out / 'train.log').open('a') as log:
            child = subprocess.Popen(command, cwd=root / 'repos/RoboActualizer', env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            status(phase='training', child_pid=child.pid)
            if child.wait():
                raise RuntimeError('Continuation training failed; inspect train.log')
        final = json.loads((train / 'checkpoints/state/latest/trainer_state.json').read_text())
        if final['global_step'] != 40000:
            raise RuntimeError('Unexpected final step')
        for step in (30000, 40000):
            for kind in ('clean', 'plus'):
                target = out / f'eval_{kind}_{step:06d}'
                target.mkdir()
                if kind == 'plus':
                    template = root / 'experiments/action_only/libero_20k_plus_280_shared_run'
                    for name in ('manifest.json', 'preflight.json'):
                        shutil.copy2(template / name, target / name)
                    (target / 'text_cache').symlink_to((template / 'text_cache').resolve())
                workers = []
                for gpu in range(8):
                    cmd = ['bash', str(root / 'scripts/run_10k_libero_probe.sh'), '--shard', str(gpu),
                           '--step', str(step), '--run', str(train), '--output', str(target)]
                    e = {**env, 'EVAL_GPU': str(gpu)}
                    if kind == 'plus':
                        cmd += ['--benchmark', 'plus', '--shards', '8', '--text-cfg-scale', '1.0']
                        e.update(EVAL_SCRIPT='eval_10k_robust.py', LIBERO_SOURCE=str(root / 'repos/LIBERO-plus-official'),
                                 LIBERO_CONFIG_PATH=str(root / 'experiments/action_only/plus_config'))
                    else:
                        e.update(EVAL_SCRIPT='eval_10k_libero_probe.py',
                                 LIBERO_SOURCE='/data/users/jiachangzhang/DNFT4RP/upstream/libero',
                                 LIBERO_CONFIG_PATH='/data/users/jiachangzhang/DNFT4RP/.libero-config')
                    with (target / f'gpu_{gpu}.log').open('a') as log:
                        workers.append(subprocess.Popen(cmd, env=e, stdin=subprocess.DEVNULL, stdout=log, stderr=log))
                status(phase='evaluating', action_step=step, benchmark=kind, worker_pids=[p.pid for p in workers])
                codes = [p.wait() for p in workers]
                if any(codes):
                    raise RuntimeError(f'Evaluation failed: {codes}')
                weights = train / f'checkpoints/weights/step_{step:06d}.pt'
                with weights.open('rb') as stream:
                    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                records = []
                for gpu in range(8):
                    report = json.loads((target / f'shard_{gpu}/report.json').read_text())
                    if report['phase'] != 'complete' or report['checkpoint_sha256'] != sha:
                        raise RuntimeError('Incomplete or mismatched evaluation')
                    records.extend(report['tasks' if kind == 'clean' else 'episodes'])
                expected = 200 if kind == 'clean' else 280
                keys = {(r['suite'], r['task_id'], r['trial']) for r in records}
                if len(records) != expected or len(keys) != expected:
                    raise RuntimeError('Missing or duplicate evaluation episodes')
                successes = sum(bool(r['success']) for r in records)
                (target / 'summary.json').write_text(json.dumps(dict(episodes=expected, successes=successes,
                    success_rate=successes/expected, records=records, checkpoint_sha256=sha), indent=2))
        status(phase='complete', finished_unix=time.time())
    except BaseException as exc:
        for process in [child, *workers]:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        status(phase='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()

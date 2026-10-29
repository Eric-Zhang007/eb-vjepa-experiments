"""One policy service per selected GPU, independent simulator groups, strict aggregation."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import signal
import hashlib
import ctypes

from elaf.evaluation.runner import episode_key


def egl_device_for_cuda(gpu):
    """EGL enumeration order can differ from the physical CUDA GPU order."""
    egl = ctypes.CDLL('libEGL.so.1')
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(
            egl.eglGetProcAddress(b'eglQueryDevicesEXT'))
    attribute = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ssize_t))(egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
    devices = (ctypes.c_void_p * 64)()
    count = ctypes.c_int()
    if not query(len(devices), devices, ctypes.byref(count)):
        raise RuntimeError('Cannot enumerate EGL devices')
    for index in range(count.value):
        ordinal = ctypes.c_ssize_t(-1)
        if attribute(devices[index], 0x323A, ctypes.byref(ordinal)) and ordinal.value == gpu:
            return index
    raise RuntimeError(f'No EGL device for physical CUDA GPU {gpu}')


def reserve_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path('/data/users/jiachangzhang/eb_vjepa'))
    p.add_argument("--policy-config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--benchmark", choices=['clean', 'plus', 'pro'], required=True)
    p.add_argument("--gpus", default='0')
    p.add_argument("--environments", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--batch-wait-ms", type=float, default=2)
    p.add_argument("--execute-steps", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus) or min(gpus) < 0:
        p.error('GPU ids must be distinct and nonnegative')
    specs = json.loads(args.manifest.read_text())
    if len({episode_key(s) for s in specs}) != len(specs):
        p.error('Duplicate episode identities in manifest')
    if len(specs) < len(gpus):
        p.error('Require at least one episode per GPU')
    if any(s['benchmark'] != args.benchmark for s in specs):
        p.error('Manifest benchmark mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob('*.py')}
    (args.output/'source_hashes.json').write_text(json.dumps(source_hashes, indent=2))
    auth = args.output / 'auth.key'
    auth.write_bytes(secrets.token_bytes(32))
    auth.chmod(0o600)
    policy_cfg = json.loads(args.policy_config.read_text())
    policy_cfg['device'] = 'cuda:0'
    policy_cfg['max_batch_size'] = max(int(policy_cfg.get('max_batch_size', 32)), args.batch_size)
    service_config = args.output / 'policy.json'
    service_config.write_text(json.dumps(policy_cfg, indent=2))
    policy_python = args.root / 'envs' / ('dream_vla_official' if policy_cfg['type'] in ('dream_vla_official', 'elaf_v2', 'dream_jepa_dualview') else ('elaf' if policy_cfg['type'] == 'elaf' else 'action_only_test')) / 'bin/python'
    dream_policy = policy_cfg['type'] in ('dream_vla_official', 'elaf_v2', 'dream_jepa_dualview')
    env_python = args.root / ('envs/dream_vla_official/bin/python' if dream_policy else 'envs/action_only_test/bin/python')
    children, logs = [], []
    started = time.perf_counter()
    try:
        jobs = []
        for index, gpu in enumerate(gpus):
            work = args.output / f'gpu_{gpu}'
            work.mkdir()
            subset = specs[index::len(gpus)]
            manifest = work / 'manifest.json'
            manifest.write_text(json.dumps(subset, indent=2))
            egl_device = egl_device_for_cuda(gpu)
            (work/'gpu_mapping.json').write_text(json.dumps(dict(cuda_gpu=gpu, egl_device=egl_device)))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(egl_device),
                       MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=str(args.root), TMPDIR=str(args.root/'cache/tmp'))
            shared = str(args.root/'envs/eb_vjepa_pretrain/lib/python3.11/site-packages')
            if policy_cfg['type'] not in ('elaf','dream_vla_official','elaf_v2','dream_jepa_dualview'):
                env['PYTHONPATH'] += ':' + shared
            port = reserve_port()
            ready = work / 'ready.json'
            log = (work/'policy.log').open('w'); logs.append(log)
            proc = subprocess.Popen([str(policy_python), '-m', 'elaf.evaluation.rpc', '--config', str(service_config),
                '--port', str(port), '--auth-file', str(auth), '--ready-file', str(ready)],
                cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(proc)
            jobs.append((gpu, work, manifest, env, port, ready, proc))
        for gpu, work, manifest, env, port, ready, service in jobs:
            deadline = time.monotonic() + 600
            while not ready.exists():
                if service.poll() is not None:
                    raise RuntimeError(f'Policy service failed; inspect {work}/policy.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Policy loading timeout')
                time.sleep(.5)
            metadata = json.loads(ready.read_text())['metadata']
            if metadata['action_space'] != 'libero_delta_ee' or metadata['action_dim'] != 7:
                raise ValueError('Policy incompatible with LIBERO action space')
            if not 1 <= args.execute_steps <= metadata['action_horizon']:
                raise ValueError('execute_steps outside trained horizon')
            config = dict(root=str(args.root), benchmark=args.benchmark, execute_steps=args.execute_steps,
                          history_length=metadata.get('history_length', 1), settle_steps=10 if dream_policy else 30,
                          gripper_convention='open_01')
            if dream_policy and args.benchmark == 'clean':
                config['max_steps_by_suite'] = dict(libero_spatial=220, libero_object=280,
                                                   libero_goal=300, libero_10=520)
            if dream_policy and args.benchmark == 'plus':
                config['plus_language_mode'] = 'canonical_clean'
            worker_config = work/'worker.json'; worker_config.write_text(json.dumps(config, indent=2))
            env = dict(env)
            # robosuite compares EGL ordinals against CUDA_VISIBLE_DEVICES,
            # although their enumeration orders can differ. Simulator workers
            # use EGL only; keep their explicit EGL selection and remove this
            # CUDA-only restriction. Policy services retain CUDA isolation.
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env['PYTHONPATH'] = str(args.root) if dream_policy else str(args.root) + ':' + shared
            env['LD_LIBRARY_PATH'] = ':'.join([str(args.root/'cache/imagemagick/root/usr/lib/x86_64-linux-gnu'),
                str(args.root/'cache/osmesa/root/usr/lib/x86_64-linux-gnu'), env.get('LD_LIBRARY_PATH','')])
            log = (work/'simulators.log').open('w'); logs.append(log)
            runner = subprocess.Popen([str(env_python), '-m', 'elaf.evaluation.runner', '--manifest', str(manifest),
                '--worker-config', str(worker_config), '--output', str(work/'results'), '--port', str(port),
                '--auth-file', str(auth), '--environments', str(args.environments), '--batch-size', str(args.batch_size),
                '--batch-wait-ms', str(args.batch_wait_ms)],
                cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(runner)
        runners = children[len(gpus):]
        while any(p.poll() is None for p in runners):
            if any(p.poll() not in (None, 0) for p in runners):
                raise RuntimeError('Simulator group failed; no complete aggregate will be published')
            time.sleep(.5)
        if any(p.returncode != 0 for p in runners):
            raise RuntimeError('Simulator group failed')
        rows, summaries = [], []
        for gpu in gpus:
            directory = args.output/f'gpu_{gpu}/results'
            summaries.append(json.loads((directory/'summary.json').read_text()))
            rows.extend(json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines())
        canonical = lambda spec: json.dumps(spec, sort_keys=True)
        if sorted(canonical(r['spec']) for r in rows) != sorted(canonical(s) for s in specs):
            raise ValueError('Duplicate, missing, or foreign episode results')
        elapsed = time.perf_counter()-started
        result = dict(complete=True, episodes=len(rows), successes=sum(r['success'] for r in rows),
                      wall_seconds=elapsed, episodes_per_hour=len(rows)*3600/elapsed,
                      includes_model_load=True, groups=summaries, gpus=gpus,
                      environments_per_gpu=args.environments, inference_batch_limit=args.batch_size)
        (args.output/'summary.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
        for process in children:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()

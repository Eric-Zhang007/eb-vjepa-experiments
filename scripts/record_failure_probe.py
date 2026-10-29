"""Replay four recorded clean failures without altering policy or execution."""
import json
import os
from pathlib import Path
import time
import numpy as np
import av
from PIL import Image, ImageDraw
from eval_10k_libero_probe import (ROOT, OmegaConf, instantiate, torch, CachedTextAdapter,
    load_dataset_stats_from_json, benchmark, get_libero_env, LIBERO_ENV_RESOLUTION,
    set_global_seed, run_single_episode)

base = ROOT / 'experiments/action_only/old10k_extend40k'
out = base / 'failure_videos_040000'
out.mkdir(exist_ok=True)
import fcntl
lock = (out / 'owner.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (out / 'status.json').exists():
    raise RuntimeError('Existing diagnostic; inspect before retrying')
status = dict(phase='loading', pid=os.getpid(), results=[])
def save():
    (out / 'status.json').write_text(json.dumps(status, indent=2))
save()
try:
    report = json.loads((base / 'eval_clean_040000/summary.json').read_text())
    selected = []
    for suite in ('libero_spatial', 'libero_goal', 'libero_10'):
        failures = [r for r in report['records'] if r['suite'] == suite and not r['success']]
        selected.extend(failures[:2] if suite == 'libero_10' else failures[:1])
    run = base / 'train'
    cfg = OmegaConf.load(run / 'config.yaml')
    cfg.model.eval_encode_fp32 = True
    cfg.EVALUATION = dict(task_suite_name='libero_spatial', replan_steps=10,
        num_steps_wait=30, num_inference_steps=4, text_cfg_scale=1.0, binarize_gripper=True)
    torch.set_num_threads(2)
    set_global_seed(239, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=torch.float32, device='cuda').eval()
    model.load_checkpoint(str(run / 'checkpoints/weights/step_040000.pt'))
    adapter = CachedTextAdapter(model, cfg.model)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run / 'dataset_stats.json')))
    for row in selected:
        status.update(phase='recording', current=row); save()
        suite = benchmark.get_benchmark_dict()[row['suite']]()
        cfg.EVALUATION.task_suite_name = row['suite']
        env, language = get_libero_env(suite.get_task(row['task_id']), LIBERO_ENV_RESOLUTION,
                                     row['seed'] - row['trial'])
        actions = []
        original_step = env.step
        def step(action):
            actions.append(np.asarray(action).copy())
            return original_step(action)
        env.step = step
        try:
            set_global_seed(row['seed'], get_worker_init_fn=False)
            success, frames = run_single_episode(env, suite.get_task_init_states(row['task_id'])[row['trial']],
                language, adapter, processor, cfg, row['trial'], 'cuda')
        finally:
            env.close()
        stem = f"{row['suite']}_{row['task_id']}_{row['trial']}"
        cameras = list(frames[0])
        with av.open(str(out / f'{stem}.mp4'), mode='w') as writer:
            stream = writer.add_stream('libx264', rate=20)
            stream.width, stream.height = 512, 256
            stream.pix_fmt = 'yuv420p'
            for frame in frames:
                vf = av.VideoFrame.from_ndarray(np.concatenate([frame[c] for c in cameras], axis=1), format='rgb24')
                for packet in stream.encode(vf):
                    writer.mux(packet)
            for packet in stream.encode():
                writer.mux(packet)
        sheet = Image.new('RGB', (1024, 4 * 280), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, index in enumerate(np.linspace(0, len(frames)-1, 8).astype(int)):
            tile = np.concatenate([frames[index][c] for c in cameras], axis=1)
            x, y = (n % 2)*512, (n // 2)*280
            sheet.paste(Image.fromarray(tile), (x, y+24))
            draw.text((x+8, y+4), f'Control step {index}', fill='black')
        sheet.save(out / f'{stem}.jpg')
        a = np.stack(actions)[30:]
        np.savez_compressed(out / f'{stem}_actions.npz', actions=a)
        delta = np.linalg.norm(np.diff(a[:, :6], axis=0), axis=1)
        boundary = (np.arange(1, len(a)) % 10 == 0)
        result = dict(source=row, replay_success=success, cameras=cameras,
            video=f'{stem}.mp4', frames=len(frames),
            mean_action_delta=float(delta.mean()),
            boundary_delta=float(delta[boundary].mean()) if boundary.any() else None,
            within_chunk_delta=float(delta[~boundary].mean()),
            gripper_switches=int(np.count_nonzero(np.diff(np.sign(a[:,6])))))
        status['results'].append(result); save()
    status['phase'] = 'complete'; save()
except BaseException as exc:
    status.update(phase='failed', error=repr(exc)); save()
    raise

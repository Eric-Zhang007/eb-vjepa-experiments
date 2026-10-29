"""Paired diagnostic pilot; preserve the production rollout and intercept env.step only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eval_10k_libero_probe import ROOT, CachedTextAdapter
from experiments.libero.eval_libero_single import run_single_episode
from experiments.libero.libero_utils import get_libero_env, LIBERO_ENV_RESOLUTION
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from Roboactualizer.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


class TraceEnv:
    def __init__(self, env, alpha, wait=30):
        self.env, self.alpha, self.wait = env, alpha, wait
        self.raw, self.executed, self.positions, self.grippers = [], [], [], []
        self.previous = None
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError('Invalid simulator action')
        applied = raw.copy()
        active = self.calls >= self.wait
        if active:
            if self.previous is not None:
                applied[:6] = self.alpha * raw[:6] + (1 - self.alpha) * self.previous[:6]
            self.previous = applied.copy()
        result = self.env.step(applied.tolist())
        self.calls += 1
        if active:
            self.raw.append(raw)
            self.executed.append(applied)
            self.positions.append(result[0]['robot0_eef_pos'].copy())
            self.grippers.append(result[0]['robot0_gripper_qpos'].copy())
        return result


def metrics(actions):
    a = np.asarray(actions)
    delta = np.diff(a[:, :6], axis=0)
    boundary = np.arange(1, len(a)) % 10 == 0
    return {
        'mean_action_delta': float(np.linalg.norm(delta, axis=1).mean()),
        'boundary_delta': float(np.linalg.norm(delta[boundary], axis=1).mean()) if boundary.any() else None,
        'within_chunk_delta': float(np.linalg.norm(delta[~boundary], axis=1).mean()),
        'mean_second_difference': float(np.linalg.norm(np.diff(a[:, :6], n=2, axis=0), axis=1).mean()),
        'gripper_switches': int(np.count_nonzero(np.diff(a[:, 6]))),
        'motion_reversal_rate': float(((a[1:, :6] * a[:-1, :6] < 0) & (np.abs(a[1:, :6]) > .05) & (np.abs(a[:-1, :6]) > .05)).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shard', type=int, required=True, choices=range(8))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    out = args.output / f'shard_{args.shard}'
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    run = Path(manifest['run'])
    checkpoint = run / 'checkpoints/weights/step_020000.pt'
    cfg = OmegaConf.load(run / 'config.yaml')
    cfg.model.eval_encode_fp32 = True
    cfg.EVALUATION = dict(task_suite_name='libero_spatial', replan_steps=10, num_steps_wait=30,
                          num_inference_steps=4, text_cfg_scale=1., binarize_gripper=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20)
    set_global_seed(239, get_worker_init_fn=False)
    model = instantiate(cfg.model, model_dtype=torch.float32, device='cuda').eval()
    model.load_checkpoint(str(checkpoint))
    adapter = CachedTextAdapter(model, cfg.model)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run / 'dataset_stats.json')))
    report = dict(phase='running', checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  alpha_definition='executed_t[:6] = alpha*raw_t[:6] + (1-alpha)*executed_previous[:6]; gripper unchanged',
                  items=[], pid=os.getpid(), started_unix=time.time())
    def save():
        temp = out / 'report.tmp'
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(out / 'report.json')
    save()
    for i, entry in enumerate(manifest['entries']):
        if i % 8 != args.shard:
            continue
        suite = benchmark.get_benchmark_dict()[entry['suite']]()
        task = suite.get_task(entry['task_id'])
        state = suite.get_task_init_states(entry['task_id'])[entry['trial']]
        cfg.EVALUATION.task_suite_name = entry['suite']
        for alpha in (1., .8, .5):
            set_global_seed(entry['seed'], get_worker_init_fn=False)
            # Original probe seeds environment per task; global RNG per trial.
            env, language = get_libero_env(task, LIBERO_ENV_RESOLUTION, entry['seed'] - entry['trial'])
            set_global_seed(entry['seed'], get_worker_init_fn=False)
            wrapped = TraceEnv(env, alpha)
            begin = time.time()
            try:
                success, frames = run_single_episode(wrapped, state, language, adapter, processor, cfg, entry['trial'], 'cuda')
                stem = f"{entry['suite']}_{entry['task_id']}_{entry['trial']}_a{alpha}"
                np.savez_compressed(out / f'{stem}.npz', raw=wrapped.raw, executed=wrapped.executed,
                                    eef_pos=wrapped.positions, gripper_qpos=wrapped.grippers)
                sheet = Image.new('RGB', (256*4, 280*4), 'white')
                draw = ImageDraw.Draw(sheet)
                for j, idx in enumerate(np.linspace(0, len(frames)-1, 8).astype(int)):
                    x, y = j % 4 * 256, j // 4 * 560
                    sheet.paste(Image.fromarray(frames[idx]['image']), (x,y))
                    sheet.paste(Image.fromarray(frames[idx]['wrist_image']), (x,y+280))
                    draw.text((x+3,y+257), f'step={idx} success={success}', fill='black')
                sheet.save(out / f'{stem}.jpg')
                item = dict(entry, alpha=alpha, success=success, steps=len(wrapped.raw),
                            raw_metrics=metrics(wrapped.raw), executed_metrics=metrics(wrapped.executed),
                            elapsed_seconds=time.time()-begin, trace=f'{stem}.npz', contact_sheet=f'{stem}.jpg')
                report['items'].append(item)
                save()
                print('RESULT ' + json.dumps(item), flush=True)
                del frames
            finally:
                env.close()
    report.update(phase='complete', finished_unix=time.time())
    save()


if __name__ == '__main__':
    main()

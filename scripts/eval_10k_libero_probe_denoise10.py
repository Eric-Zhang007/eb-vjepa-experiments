"""Forty-task, one-init-state LIBERO probe, isolated from production training."""
from collections import deque
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
sys.path.insert(0, str(ROOT / "repos/RoboActualizer"))
from experiments.libero.eval_libero_single import run_single_episode
from experiments.libero.libero_utils import get_libero_env, LIBERO_ENV_RESOLUTION
from experiments.libero.joint_fm_adapter import LiberoJointPredictFMAdapter
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from Roboactualizer.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


class CachedTextAdapter(LiberoJointPredictFMAdapter):
    """Use the existing real umT5 embeddings instead of reloading umT5 per task."""
    def __init__(self, model, cfg):
        self.model = model
        self.device = "cuda"
        self.torch_dtype = torch.bfloat16
        self.pixel_dtype = torch.float32
        self.clip_len = int(cfg.clip_len)
        self.frame_interval = int(cfg.frame_interval)
        self.context_len = int(cfg.tokenizer_max_len)
        self.accel = {k: False for k in (
            "use_kv_cache", "use_cuda_graph", "use_adaln_cache", "use_cross_kv_cache")}
        self.frame_buffer = deque(maxlen=(self.clip_len - 1) * self.frame_interval + 1)
        self.cached = {}

    def _encode_instruction(self, prompt):
        if prompt not in self.cached:
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            name = f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
            candidates = [ROOT / "experiments/action_only/libero_10k_egl_200/text_cache" / name,
                          ROOT / "experiments/action_only/text_cache" / name]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if "prompt" in payload:
                assert payload["prompt"] == prompt
            context = payload["context"].clone()
            mask = payload["mask"].bool()
            context[~mask] = 0
            assert context.shape == (128, 4096) and mask.any() and torch.isfinite(context).all()
            self.cached[prompt] = (context[None].cuda().to(self.torch_dtype), mask[None].cuda())
        return self.cached[prompt]

    def infer_action(self, **kwargs):
        result = super().infer_action(**kwargs)
        assert torch.isfinite(result["action"]).all()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True, choices=range(8))
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--run", type=Path, default=ROOT / "experiments/action_only/train_final")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = (args.output or ROOT / f"experiments/action_only/libero_{args.step // 1000}k_egl_200") / f"shard_{args.shard}"
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "probe.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report_path = out / "report.json"
    if report_path.exists():
        raise RuntimeError("Existing probe report: do not overwrite or duplicate")
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.15)
    run = args.run
    checkpoint = run / f"checkpoints/weights/step_{args.step:06d}.pt"
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.model.eval_encode_fp32 = True
    cfg.EVALUATION = {
        "task_suite_name": "libero_spatial", "replan_steps": 10, "num_steps_wait": 30,
        "num_inference_steps": 10, "text_cfg_scale": 1.0, "binarize_gripper": True,
    }
    set_global_seed(239, get_worker_init_fn=False)
    report = {
        "phase": "loading", "pid": os.getpid(), "checkpoint": str(checkpoint),
        "checkpoint_step": args.step,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "seed": 239, "initial_state_indices": list(range(5)), "trials_per_task": 5,
        "planned_tasks": 5, "planned_episodes": 25, "shard": args.shard,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "rendering": os.environ["MUJOCO_GL"],
        "protocol": "Exploratory clean LIBERO 5 trials/task, NOT standard 50 trials/task",
        "inference_steps": 10, "cfg": 1.0, "replan_steps": 10,
        "text": "Real cached umT5, exact prompt hash; no fallback",
        "started_unix": time.time(), "tasks": [],
    }
    def save():
        temp = report_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(report_path)
    save()
    try:
        model = instantiate(cfg.model, model_dtype=torch.float32, device="cuda").eval()
        model.load_checkpoint(str(checkpoint))
        adapter = CachedTextAdapter(model, cfg.model)
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run / "dataset_stats.json")))
        report["phase"] = "running"
        for suite_idx, suite_name in enumerate(("libero_spatial", "libero_object", "libero_goal", "libero_10")):
            suite = benchmark.get_benchmark_dict()[suite_name]()
            cfg.EVALUATION.task_suite_name = suite_name
            for task_id in range(suite.n_tasks):
                global_task = suite_idx * 10 + task_id
                if global_task % 8 != args.shard:
                    continue
                seed = 239 + global_task * 5
                task = suite.get_task(task_id)
                states = suite.get_task_init_states(task_id)
                env, bddl_language = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
                language = task.language
                assert isinstance(language, str) and language.strip()
                begin = time.monotonic()
                try:
                    from OpenGL import GL
                    renderer = GL.glGetString(GL.GL_RENDERER)
                    if renderer is not None:
                        report["gl_renderer"] = renderer.decode()
                        assert "NVIDIA" in report["gl_renderer"] or "RTX" in report["gl_renderer"]
                    for trial in range(5):
                        set_global_seed(seed + trial, get_worker_init_fn=False)
                        success, frames = run_single_episode(
                            env, states[trial], language, adapter, processor, cfg, trial, "cuda")
                        for label, frame in (("first", frames[0]), ("last", frames[-1])):
                            for camera, image in frame.items():
                                assert image.shape == (256, 256, 3) and image.std() > 0
                                Image.fromarray(image).save(
                                    out / f"{suite_name}_{task_id}_{trial}_{label}_{camera}.png")
                        item = {"suite": suite_name, "task_id": task_id, "language": language, "bddl_language": bddl_language,
                                "input_instruction": language, "instruction_source": "task.language",
                                "trial": trial, "success": success, "seed": seed + trial,
                                "recorded_steps": len(frames), "elapsed_seconds": time.monotonic() - begin}
                        report["tasks"].append(item)
                        save()
                        print("TASK_RESULT " + json.dumps(item), flush=True)
                        del frames
                finally:
                    env.close()
        report["phase"] = "complete"
        report["successes"] = sum(x["success"] for x in report["tasks"])
        report["success_rate"] = report["successes"] / len(report["tasks"])
        report["finished_unix"] = time.time()
        save()
        print("PROBE_COMPLETE " + str(report["successes"]) + "/25", flush=True)
    except Exception as exc:
        report.update(phase="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()

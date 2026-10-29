"""Bounded actual-policy rollout plumbing check; never a benchmark score."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from experiments.libero.eval_libero_single import (
    _normalize_proprio, _extract_sim_state, _update_model_history, _denormalize_action,
)
from experiments.libero.libero_utils import get_libero_env, get_libero_dummy_action
from experiments.libero.joint_fm_adapter import LiberoJointPredictFMAdapter
from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from libero.libero import benchmark, get_libero_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.run / "config.yaml")
    model = instantiate(cfg.model, model_dtype=torch.float32, device="cuda")
    checkpoint = args.run / "checkpoints/weights/step_000008.pt"
    model.load_checkpoint(str(checkpoint))
    model.eval()
    cfg.model.eval_encode_fp32 = True
    for flag in ("use_kv_cache", "use_cuda_graph", "use_adaln_cache", "use_cross_kv_cache"):
        cfg.model[flag] = False
    adapter = LiberoJointPredictFMAdapter(model, cfg.model, "cuda", torch.bfloat16)
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(args.run / "dataset_stats.json")))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(0)
    env, language = get_libero_env(task, 256, 239)
    try:
        env.reset()
        initial = suite.get_task_init_states(0)
        obs = env.set_init_state(initial[0])
        adapter.reset_history()
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())
            _update_model_history(adapter, obs, processor, "cuda")
        actions = []
        for _ in range(3):
            with torch.no_grad():
                prediction = adapter.infer_action(
                    prompt=DEFAULT_PROMPT.format(task=language),
                    proprio=_normalize_proprio(_extract_sim_state(obs), processor),
                    num_inference_steps=4, text_cfg_scale=1.0)["action"]
            action = _denormalize_action(prediction, processor)[0, 0]
            action[-1] = np.sign(1 - 2 * action[-1])
            assert action.shape == (7,) and np.isfinite(action).all()
            obs, _, done, _ = env.step(action.tolist())
            actions.append(action.tolist())
            _update_model_history(adapter, obs, processor, "cuda")
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                assert obs[key].shape == (256, 256, 3) and obs[key].std() > 0
            if done:
                break
        args.output.write_text(json.dumps({
            "status": "PASS", "suite": args.suite, "language": language,
            "checkpoint": str(checkpoint), "policy_steps": len(actions), "actions": actions,
            "benchmark_score": None, "purpose": "8-step-trained policy plumbing only",
        }, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""Validate NVIDIA EGL and encode exact BDDL instructions into an isolated cache."""
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from OpenGL import GL

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
sys.path.insert(0, str(ROOT / "repos/RoboActualizer"))
from experiments.libero.libero_utils import get_libero_env, get_libero_task_description
from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from Roboactualizer.models.roboactualizer.runtime import load_text_encoder_components
from libero.libero import benchmark

torch.set_num_threads(2)
out = ROOT / "experiments/action_only/libero_10k_egl_200"
cache = out / "text_cache"
cache.mkdir(parents=True, exist_ok=True)
prompts = []
for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    suite = benchmark.get_benchmark_dict()[name]()
    for i in range(suite.n_tasks):
        language = get_libero_task_description(suite.get_task(i))
        prompts.append({"suite": name, "task": i, "language": language,
                        "prompt": DEFAULT_PROMPT.format(task=language)})
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
env, _ = get_libero_env(suite.get_task(0), 256, 239)
try:
    obs = env.reset()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    vendor = GL.glGetString(GL.GL_VENDOR).decode()
    assert "NVIDIA" in vendor, (renderer, vendor)
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        assert obs[key].shape == (256, 256, 3) and np.std(obs[key]) > 0
    print(json.dumps({"renderer": renderer, "vendor": vendor}), flush=True)
finally:
    env.close()
text_encoder, tokenizer = load_text_encoder_components(
    model_id="Wan-AI/Wan2.2-TI2V-5B", tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
    context_len=128, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    for item in prompts:
        prompt = item["prompt"]
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        mask = mask.cuda().bool()
        context = text_encoder(ids.cuda(), mask)
        context[~mask] = 0
        assert torch.isfinite(context).all() and mask.any()
        name = hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
        torch.save({"context": context[0].cpu(), "mask": mask[0].cpu(), "prompt": prompt}, cache / name)
        print("TEXT_OK", item["suite"], item["task"], flush=True)
(out / "preflight.json").write_text(json.dumps({
    "status": "PASS", "gl_renderer": renderer, "gl_vendor": vendor,
    "prompts": prompts, "source": "Exact simulator BDDL language; real pretrained umT5",
}, indent=2))
del text_encoder
gc.collect()
torch.cuda.empty_cache()

"""Isolated, manifest-defined LIBERO-Plus/Pro evaluation of the step-10k policy."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from eval_10k_libero_probe import CachedTextAdapter, ROOT
from experiments.libero.eval_libero_single import run_single_episode
from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from Roboactualizer.models.roboactualizer.runtime import load_text_encoder_components
from Roboactualizer.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import get_problem_info

BASES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORIES = ("Camera Viewpoints", "Robot Initial States", "Language Instructions",
              "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout")


class LegacyBinaryNumpy:
    """Process-local compatibility for the upstream Plus PNG byte decoder."""
    def __getattr__(self, name):
        if name == "float_":
            return np.float64
        return getattr(np, name)

    def fromstring(self, string, dtype=float, count=-1, sep="", **kwargs):
        if sep == "":
            return np.frombuffer(string, dtype=dtype, count=count, **kwargs)
        return np.fromstring(string, dtype=dtype, count=count, sep=sep, **kwargs)


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def task_language(task):
    # Plus encodes camera/robot/noise parameters in a virtual filename. The
    # environment resolves the BDDL prefix while retaining those parameters.
    name = task.bddl_file.split("_view_")[0] + ".bddl" if "_view_" in task.bddl_file else task.bddl_file
    path = Path(get_libero_path("bddl_files")) / task.problem_folder / name
    language = get_problem_info(str(path))["language_instruction"]
    assert isinstance(language, str) and language.strip()
    return language


def environment(task, seed):
    path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(path), camera_heights=256, camera_widths=256)
    env.seed(seed)
    return env


class TextAdapter(CachedTextAdapter):
    def __init__(self, model, cfg, cache, instruction_mode="original"):
        super().__init__(model, cfg)
        self.cache = cache
        self.instruction_mode = instruction_mode

    def _encode_instruction(self, prompt):
        if self.instruction_mode == "empty":
            # Bypass even DEFAULT_PROMPT's boilerplate, not only its task slot.
            prompt = ""
        if prompt not in self.cached:
            name = hashlib.sha256(prompt.encode()).hexdigest() + f".t5_len128.{self.text_cache_suffix}.pt"
            payload = torch.load(self.cache / name, map_location="cpu", weights_only=True)
            assert payload["prompt"] == prompt
            context, mask = payload["context"].clone(), payload["mask"].bool()
            assert context.shape == (128, 4096) and mask.any() and torch.isfinite(context).all()
            context[~mask] = 0
            self.cached[prompt] = context[None].cuda().to(self.torch_dtype), mask[None].cuda()
        return self.cached[prompt]


def prepare(kind, out):
    lock = (out / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "preflight.json").exists():
        raise RuntimeError("Preflight already exists; refusing to overwrite")
    source = Path(os.environ["LIBERO_SOURCE"])
    rng = random.Random(239)
    suite_cache = {}
    entries = []
    if kind == "plus":
        classification = source / "libero/libero/benchmark/task_classification.json"
        classified = json.loads(classification.read_text())
        for base in BASES:
            suite_cache[base] = benchmark.get_benchmark_dict()[base]()
            for cat in CATEGORIES:
                pool = sorted((e for e in classified[base] if e["category"] == cat), key=lambda e: e["id"])
                for e in sorted(rng.sample(pool, 10), key=lambda e: e["id"]):
                    idx = int(e["id"]) - 1
                    task = suite_cache[base].get_task(idx)
                    assert task.name == e["name"]
                    entries.append({"suite": base, "task_id": idx, "category": cat, "trials": 1})
    else:
        for base in BASES:
            for cat in ("lan", "object", "swap", "task"):
                name = base + "_" + cat
                suite_cache[name] = benchmark.get_benchmark_dict()[name]()
                assert suite_cache[name].n_tasks > 0
                entries.extend({"suite": name, "task_id": i, "category": cat, "trials": 5}
                               for i in range(suite_cache[name].n_tasks))
    # Validate every selected BDDL and init-state mapping before GPU rollout.
    for index, entry in enumerate(entries):
        suite = suite_cache[entry["suite"]]
        task = suite.get_task(entry["task_id"])
        states = suite.get_task_init_states(entry["task_id"])
        assert len(states) >= entry["trials"], entry
        assert np.isfinite(np.asarray(states)).all(), entry
        entry.update(index=index, name=task.name, language=task_language(task))
    manifest = {
        "benchmark": kind, "source": str(source),
        "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "seed": 239, "entries": entries, "episodes": sum(e["trials"] for e in entries),
        "protocol": "Exploratory stratified subset" if kind == "plus" else "Four installed official perturbation families",
        "excluded_pro_dimension": "environment (no verified pre-generated assets)" if kind == "pro" else None,
        "horizons": "400 for spatial/object/goal; 700 for libero_10 (same as clean probe)",
    }
    atomic(out / "manifest.json", manifest)
    render_checks = []
    # One real reset per perturbation family; all tasks are still checked by workers.
    from OpenGL import GL
    for cat in dict.fromkeys(e["category"] for e in entries):
        e = next(e for e in entries if e["category"] == cat)
        suite = suite_cache[e["suite"]]
        task = suite.get_task(e["task_id"])
        set_global_seed(239, get_worker_init_fn=False)
        env = environment(task, 239)
        try:
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(e["task_id"])[0])
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            assert "NVIDIA" in GL.glGetString(GL.GL_VENDOR).decode()
            for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                image = obs[camera]
                assert image.shape == (256, 256, 3) and np.std(image) > 0
                Image.fromarray(image).save(out / f"preflight_{cat.replace(' ', '_')}_{camera}.png")
            render_checks.append({"category": cat, "renderer": renderer, "task": e["name"]})
            print("RENDER_PASS", cat, flush=True)
        finally:
            env.close()
    cache = out / "text_cache"
    cache.mkdir(exist_ok=True)
    prompts = sorted({DEFAULT_PROMPT.format(task=e["language"]) for e in entries})
    encoder, tokenizer = load_text_encoder_components(
        model_id="Wan-AI/Wan2.2-TI2V-5B", tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        context_len=128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        for offset in range(0, len(prompts), 4):
            batch = prompts[offset:offset+4]
            ids, mask = tokenizer(batch, return_mask=True, add_special_tokens=True)
            mask = mask.cuda().bool()
            context = encoder(ids.cuda(), mask)
            context[~mask] = 0
            assert torch.isfinite(context).all()
            for i, prompt in enumerate(batch):
                name = hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt"
                torch.save({"context": context[i].cpu(), "mask": mask[i].cpu(), "prompt": prompt}, cache / name)
            print("TEXT", min(offset+4, len(prompts)), len(prompts), flush=True)
    atomic(out / "preflight.json", {"status": "PASS", "render_checks": render_checks,
                                   "unique_prompts": len(prompts),
                                   "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()})


def worker(kind, out, shard, step=10000, shards=8, memory_fraction=0.20,
           instruction_mode="original", text_cfg_scale=1.0, run=None):
    if text_cfg_scale != 1.0 and instruction_mode != "original":
        raise ValueError("CFG comparison requires original instructions")
    local = out / f"shard_{shard}"
    local.mkdir(exist_ok=True)
    lock = (local / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = local / "report.json"
    if path.exists():
        raise RuntimeError("Existing worker report; refusing duplicate")
    manifest = json.loads((out / "manifest.json").read_text())
    preflight = json.loads((out / "preflight.json").read_text())
    assert preflight["status"] == "PASS"
    assert preflight["manifest_sha256"] == hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    assert manifest.get("instruction_mode", "original") == instruction_mode
    if instruction_mode in ("empty", "swapped"):
        assert all(e["category"] == "Language Instructions" for e in manifest["entries"])
    if instruction_mode == "swapped":
        by_index = {e["index"]: e for e in manifest["entries"]}
        assert len(by_index) == len(manifest["entries"])
        assert sorted(e["donor_index"] for e in manifest["entries"]) == sorted(by_index)
        for e in manifest["entries"]:
            donor = by_index[e["donor_index"]]
            assert donor["suite"] == e["suite"] and donor["index"] != e["index"]
            assert "_language_" in donor["name"] and "_language_" in e["name"]
            assert donor["name"].split("_language_")[0] != e["name"].split("_language_")[0]
            assert e["swapped_language"] == donor["language"] != e["language"]
    if not 0 <= shard < shards:
        raise ValueError("Invalid shard")
    entries = [e for e in manifest["entries"] if e["index"] % shards == shard]
    run = run or ROOT / "experiments/action_only/train_final"
    ckpt = run / f"checkpoints/weights/step_{step:06d}.pt"
    report = {"phase": "loading", "pid": os.getpid(), "benchmark": kind, "shard": shard,
              "planned_episodes": sum(e["trials"] for e in entries), "episodes": [],
              "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
              "checkpoint_step": step, "shards": shards,
              "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
              "memory_fraction": memory_fraction, "instruction_mode": instruction_mode,
              "text_cfg_scale": text_cfg_scale}
    atomic(path, report)
    try:
        cfg = OmegaConf.load(run / "config.yaml")
        cfg.model.eval_encode_fp32 = True
        cfg.EVALUATION = {"task_suite_name": "libero_spatial", "num_steps_wait": 30,
                          "replan_steps": int(cfg.model.get('evaluation_replan_steps',10)), "num_inference_steps": int(cfg.model.get('evaluation_inference_steps',4)),
                          "text_cfg_scale": text_cfg_scale, "binarize_gripper": True}
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        model = instantiate(cfg.model, model_dtype=torch.float32, device="cuda").eval()
        model.load_checkpoint(str(ckpt))
        adapter = TextAdapter(model, cfg.model, out / "text_cache", instruction_mode)
        if text_cfg_scale != 1.0:
            null_prompt = DEFAULT_PROMPT.format(task="")
            null_path = out / "text_cache" / (
                hashlib.sha256(null_prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt")
            report["unconditional_prompt"] = null_prompt
            report["unconditional_cache_sha256"] = hashlib.sha256(null_path.read_bytes()).hexdigest()
        if instruction_mode == "empty":
            empty_context, empty_mask = adapter._encode_instruction("must not reach the encoder")
            assert list(adapter.cached) == [""]
            assert empty_mask.any() and torch.isfinite(empty_context).all()
            report["empty_text_cache_sha256"] = hashlib.sha256(
                (out / "text_cache" / (hashlib.sha256(b"").hexdigest() +
                 ".t5_len128.wan22ti2v5b.pt")).read_bytes()).hexdigest()
            report["text_encoder_input"] = ""
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(run / "dataset_stats.json")))
        report["phase"] = "running"
        atomic(path, report)
        suites = {}
        for e in entries:
            if e["suite"] not in suites:
                suites[e["suite"]] = benchmark.get_benchmark_dict()[e["suite"]]()
            suite = suites[e["suite"]]
            task = suite.get_task(e["task_id"])
            assert task.name == e["name"] and task_language(task) == e["language"]
            instruction = ("" if instruction_mode == "empty" else
                           e["swapped_language"] if instruction_mode == "swapped" else e["language"])
            if instruction_mode == "swapped":
                prompt = DEFAULT_PROMPT.format(task=instruction)
                cache_path = out / "text_cache" / (
                    hashlib.sha256(prompt.encode()).hexdigest() + ".t5_len128.wan22ti2v5b.pt")
                assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == e["text_cache_sha256"]
            states = suite.get_task_init_states(e["task_id"])
            cfg.EVALUATION.task_suite_name = e["suite"]
            seed = 239 + e["index"] * 5
            env = environment(task, seed)
            try:
                for trial in range(e["trials"]):
                    set_global_seed(seed+trial, get_worker_init_fn=False)
                    start = time.monotonic()
                    success, frames = run_single_episode(
                        env, states[trial], instruction,
                        adapter, processor, cfg, trial, "cuda")
                    if instruction_mode == "swapped":
                        assert DEFAULT_PROMPT.format(task=instruction) in adapter.cached
                    if text_cfg_scale != 1.0:
                        assert null_prompt in adapter.cached
                    for label, frame in (("first", frames[0]), ("last", frames[-1])):
                        for camera, image in frame.items():
                            assert image.shape == (256, 256, 3) and image.std() > 0
                            Image.fromarray(image).save(local / f"{e['index']}_{trial}_{label}_{camera}.png")
                    row = {**e, "trial": trial, "seed": seed+trial, "success": success,
                           "steps": len(frames), "seconds": time.monotonic()-start,
                           "instruction_mode": instruction_mode,
                           "text_cfg_scale": text_cfg_scale,
                           "input_instruction": instruction}
                    report["episodes"].append(row)
                    report["max_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    report["max_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    atomic(path, report)
                    print("RESULT " + json.dumps(row), flush=True)
                    del frames
            finally:
                env.close()
        report["phase"] = "complete"
        atomic(path, report)
    except Exception as exc:
        report.update(phase="failed", error=repr(exc))
        atomic(path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=("plus", "pro"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--shard", type=int, choices=range(8))
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--memory-fraction", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--instruction-mode", choices=("original", "empty", "swapped"), default="original")
    parser.add_argument("--text-cfg-scale", type=float, choices=(1.0, 1.5), default=1.0)
    args = parser.parse_args()
    if args.benchmark == "plus":
        from libero.libero.envs import env_wrapper
        env_wrapper.np = LegacyBinaryNumpy()
    torch.set_num_threads(2)
    output = args.output or ROOT / f"experiments/action_only/libero_{args.step // 1000}k_{args.benchmark}"
    output.mkdir(exist_ok=True)
    if args.prepare:
        prepare(args.benchmark, output)
    else:
        assert args.shard is not None
        worker(args.benchmark, output, args.shard, args.step, args.shards,
               args.memory_fraction, args.instruction_mode, args.text_cfg_scale, args.run)

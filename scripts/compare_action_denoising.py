"""Bounded paired diagnostic; never changes a training run or its caches."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from Roboactualizer.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from Roboactualizer.utils import misc


def main():
    root = Path("/data/users/jiachangzhang/eb_vjepa")
    run = root / "experiments/action_only/train_final"
    out = root / "experiments/action_only/paired_denoising_010000_v1"
    out.mkdir(exist_ok=False)
    misc.register_work_dir(out)
    torch.set_num_threads(2)
    torch.manual_seed(239)
    torch.cuda.set_per_process_memory_fraction(0.20)
    cfg = OmegaConf.load(run / "config.yaml")
    dcfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    dcfg.rgb_latent_cache_dir = None
    dcfg.pretrained_norm_stats = str(run / "dataset_stats.json")
    dataset = instantiate(dcfg)
    base_processor = dataset.lerobot_dataset.processor
    base_norm = base_processor.normalizer.normalizers
    rng = np.random.default_rng(239)
    indices = sorted(rng.choice(len(dataset), 64, replace=False).tolist())
    samples = []
    for i in indices:
        sample = dataset[i]
        assert sample["action"].abs().max() < 4.99
        assert sample["proprio"].abs().max() < 4.99
        samples.append({
            "index": i, "prompt": sample["prompt"],
            "pixels": sample["video"][0],
            "raw_action": base_norm["action"]["default"].backward(sample["action"][:16]),
            "raw_proprio": base_norm["state"]["default"].backward(sample["proprio"][0]),
            "mask": sample["action_is_pad"][:16],
            "context": sample["context"], "text_mask": sample["context_mask"],
        })
    del dataset
    released = root / "checkpoints/roboactualizer/libero_dits"
    sources = [
        ("ctp_action_only_010000", run / "checkpoints/weights/step_010000.pt",
         run / "dataset_stats.json"),
        ("released_joint_173580", released / "step_173580.pt", released / "dataset_stats.json"),
    ]
    report = {
        "purpose": "Paired teacher-forced action denoising, NOT rollout or matched-budget ablation",
        "samples": 64, "noise_draws_per_sample": 4, "seed": 239,
        "indices": indices, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset": OmegaConf.to_container(dcfg, resolve=True),
        "precision": "fp32 frozen encoder, fp32 policy with bf16 autocast",
        "sample_scope": "Current selected LIBERO training data; not held-out",
        "models": {},
    }
    for tag, path, stats_path in sources:
        mcfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        official = tag.startswith("released")
        if official:
            mcfg.action_only = False
            mcfg.future_size = 4
            mcfg.image_predict_weight = 1.0
            mcfg.latent_norm_stats_path = None
            mcfg.backbone.ckpt_path = str(
                root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt")
        model = instantiate(mcfg, model_dtype=torch.float32, device="cuda").eval()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model"]
        expected = {k for k in model.state_dict() if not k.startswith("backbone.")}
        assert set(state) == expected, (sorted(expected-set(state)), sorted(set(state)-expected))
        merged = {k: v for k, v in model.state_dict().items() if k.startswith("backbone.")}
        merged.update(state)
        model.load_state_dict(merged, strict=True)
        del merged, state, payload
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        norms = processor.normalizer.normalizers
        results = []
        equivalence = None
        with torch.inference_mode():
            for j, sample in enumerate(samples):
                a = norms["action"]["default"].forward(sample["raw_action"])[None].cuda()
                p = norms["state"]["default"].forward(sample["raw_proprio"])[None].cuda()
                text = sample["context"][None].cuda()
                mask = sample["text_mask"][None].cuda()
                z = model.backbone.encode_clip_last_group(sample["pixels"][None].cuda().float())
                z = model._normalize_latent(z.to(model.rgb_latent_dtype).float())
                valid = (~sample["mask"]).float().cuda()
                scale = norms["action"]["default"].scale.cuda()
                for draw in range(4):
                    torch.manual_seed(239 + sample["index"] * 4 + draw)
                    sigma, timestep = model.action_scheduler.sample_train(1, "cuda")
                    noise = torch.randn_like(a)
                    noisy = model.action_scheduler.add_noise(a, noise, sigma)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ic, cm = model.image_context_builder(text, mask, p)
                        ac, _ = model.action_context_builder(text, mask, p)
                        pred = model._action_only_forward(z, noisy, ic, ac, cm, timestep)
                        if official and equivalence is None:
                            future = torch.zeros(1, 4, z.shape[1], z.shape[2], device="cuda")
                            _, joint = model._joint_forward(
                                z, future, noisy, ic, ac, cm, timestep, timestep)
                            equivalence = float((joint.float()-pred.float()).abs().max())
                            assert equivalence < 0.02, equivalence
                    err = pred.float() - a
                    v_err = err / sigma.clamp_min(model.action_scheduler.sigma_clip_min)
                    def reduce(x):
                        return float((x.square().mean(-1)[0] * valid).sum() / valid.sum().clamp_min(1))
                    row = {"index": sample["index"], "draw": draw, "sigma": float(sigma.item()),
                           "native_velocity_mse": reduce(v_err),
                           "raw_action_denoising_mse": reduce(err / scale)}
                    assert np.isfinite(list(row.values())).all()
                    results.append(row)
                    time.sleep(0.10)
                if j % 16 == 0:
                    print(json.dumps({"tag": tag, "samples_done": j+1}), flush=True)
        item = {
            "checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "strict_nonbackbone_load": True, "stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
            "joint_vs_action_forward_max_abs": equivalence,
            "action_scale": norms["action"]["default"].scale.tolist(),
            "action_offset": norms["action"]["default"].offset.tolist(),
            "rows": results,
        }
        for key in ("native_velocity_mse", "raw_action_denoising_mse"):
            values = np.array([np.mean([r[key] for r in results if r["index"] == i]) for i in indices])
            item[key] = {"mean": float(values.mean()), "median_per_sample": float(np.median(values))}
        report["models"][tag] = item
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in item.items() if k != "rows"}), flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

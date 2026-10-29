"""Fixed-input, label-free diagnostic; retrieval is not robot-control quality."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from app.vjepa_2_1.models.vision_transformer import vit_large
from data import decode_clip
from model import clean_state
from diagnose_features import stats


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def variants(array):
    x = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    x = F.interpolate(x, size=(round(h * 256 / min(h, w)), round(w * 256 / min(h, w))),
                      mode="bilinear", align_corners=False)
    top, left = (x.shape[-2] - 256) // 2, (x.shape[-1] - 256) // 2
    base = x[:, :, top:top + 256, left:left + 256] / 255
    # One fixed, temporally consistent crop/exposure perturbation for every model.
    aug = F.interpolate(base[:, :, 16:240, 8:232], size=(256, 256),
                        mode="bilinear", align_corners=False)
    aug = (aug * 0.85 + 0.025).clamp(0, 1)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None]
    std = torch.tensor([.229, .224, .225])[None, :, None, None]
    return [((z - mean) / std).permute(1, 0, 2, 3).unsqueeze(0) for z in (base, aug)]


def retrieval(query, gallery, positives):
    sim = F.normalize(query.double(), dim=-1) @ F.normalize(gallery.double(), dim=-1).T
    targets = torch.tensor(positives)
    correct_sim = sim[torch.arange(len(query)), targets]
    ranks = (sim > correct_sim[:, None]).sum(1) + 1
    ties = (sim == correct_sim[:, None]).sum(1)
    # Tied impostors count against the positive: collapsed features cannot pass.
    ranks = ranks + ties - 1
    negative = sim.clone()
    negative[torch.arange(len(query)), targets] = -torch.inf
    margins = correct_sim - negative.max(1).values
    return {"queries": len(query), "gallery": len(gallery),
            "top1": float((ranks <= 1).double().mean()),
            "top5": float((ranks <= 5).double().mean()),
            "mean_reciprocal_rank": float((1 / ranks.double()).mean()),
            "positive_cosine_mean": float(correct_sim.mean()),
            "hardest_negative_margin_mean": float(margins.mean()),
            "ranks": ranks.tolist(), "margins": margins.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = root / "experiments/ctp_full_72h/compare_v1_30k_full_60k"
    out.mkdir(exist_ok=True)
    lock = (out / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "status.json").exists():
        raise RuntimeError("Existing comparison; refusing duplicate")
    state = {"phase": "starting", "pid": os.getpid(), "started_unix": time.time()}
    atomic(out / "status.json", state)
    torch.set_num_threads(2)
    torch.manual_seed(239)
    try:
        free, total = torch.cuda.mem_get_info()
        if free < 4.5 * 1024**3:
            raise RuntimeError("Initial free memory below 4.5 GiB")
        torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 / total)
        clip_path = root / "experiments/ctp_full_72h/diagnostics_030000/fixed_clips.json"
        clips = json.loads(clip_path.read_text())
        assert len(clips) == 144
        atomic(out / "fixed_clips.json", clips)
        old_records = [json.loads(l) for l in
                       (root / "experiments/ctp_v1/episodes_available.jsonl").open()]
        seen_episodes = {r["episode"] for r in old_records}
        seen_pairs = {(r["episode"], r["camera"]) for r in old_records}
        sources = {
            "official": root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt",
            "old_v1_30k": root / "experiments/ctp_v1/embodied_vjepa_encoder_030000.pt",
            "full_60k": root / "experiments/ctp_full_72h/embodied_vjepa_encoder_060000.pt",
        }
        report = {"scope": "Training-distribution label-free diagnostics, not semantic/control evaluation",
                  "clips_sha256": digest(clip_path), "clips": len(clips),
                  "episodes": len({c["episode"] for c in clips}),
                  "old_training_episode_overlap": len({c["episode"] for c in clips} & seen_episodes),
                  "old_training_episode_view_overlap": sum(
                      (c["episode"], c["camera"]) in seen_pairs for c in clips),
                  "augmentation": "Center256 -> crop rows16:240 cols8:232 -> resize256; pixels*0.85+0.025",
                  "retrieval": "Augmented query against 48 clean clips of the same camera; match episode",
                  "cross_view": "Same episode midpoint in another camera, not frame-synchronized; may use scene identity",
                  "models": {}}
        all_features = {}
        ids = {v: [i for i, c in enumerate(clips) if c["camera"] == v] for v in range(3)}
        assert all(len(xs) == 48 for xs in ids.values())
        for tag, path in sources.items():
            model = vit_large(img_size=256, num_frames=16, tubelet_size=2, patch_size=16,
                              use_rope=True, use_sdpa=True, img_temporal_dim_size=1,
                              interpolate_rope=True, n_output_distillation=1)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            model.load_state_dict(clean_state(checkpoint["ema_encoder"]), strict=True)
            del checkpoint
            model = model.eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
            clean, augmented = [], []
            for i, c in enumerate(clips):
                if torch.cuda.mem_get_info()[0] < 2.5 * 1024**3:
                    raise RuntimeError("Runtime free-memory floor reached")
                array, _ = decode_clip(c["path"], c["clip_start"], c["end"])
                with torch.inference_mode():
                    for x, target in zip(variants(array), (clean, augmented)):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            features = model(x.to("cuda", dtype=torch.bfloat16))
                        pooled = features.float().mean(1).cpu()[0]
                        assert torch.isfinite(pooled).all()
                        target.append(pooled)
                        del features, x
                if i % 16 == 0:
                    state.update(phase="extracting", model=tag, clips=i + 1,
                                 max_reserved_bytes=torch.cuda.max_memory_reserved())
                    atomic(out / "status.json", state)
                    print(json.dumps(state), flush=True)
                time.sleep(0.15)
            clean, augmented = torch.stack(clean), torch.stack(augmented)
            torch.save({"clean": clean, "augmented": augmented}, out / f"{tag}_features.pt")
            all_features[tag] = clean, augmented
            result = {"checkpoint": str(path), "sha256": digest(path), "strict_load": True,
                      "views": {}, "cross_view": {}}
            for v, indexes in ids.items():
                result["views"][str(v)] = {
                    "stats": stats(clean[indexes]),
                    "augmentation_retrieval": retrieval(augmented[indexes], clean[indexes], list(range(48)))}
            for v in range(3):
                for w in range(3):
                    if v == w:
                        continue
                    gallery_map = {clips[i]["episode"]: j for j, i in enumerate(ids[w])}
                    positives = [gallery_map[clips[i]["episode"]] for i in ids[v]]
                    result["cross_view"][f"{v}_to_{w}"] = retrieval(clean[ids[v]], clean[ids[w]], positives)
            report["models"][tag] = result
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report["old_vs_full"] = {}
        rng = np.random.default_rng(239)
        for v, indexes in ids.items():
            a, b = [all_features[tag][0][indexes].double() for tag in ("old_v1_30k", "full_60k")]
            cosine = float(F.cosine_similarity(a, b).mean())
            a, b = a - a.mean(0), b - b.mean(0)
            k, l = a @ a.T, b @ b.T
            ranks = [np.asarray(report["models"][tag]["views"][str(v)]["augmentation_retrieval"]["ranks"])
                     for tag in ("old_v1_30k", "full_60k")]
            delta = (ranks[1] == 1).astype(float) - (ranks[0] == 1)
            bootstrap = delta[rng.integers(0, 48, (5000, 48))].mean(1)
            report["old_vs_full"][str(v)] = {
                "linear_cka": float((k * l).sum() / (k.norm() * l.norm()).clamp_min(1e-30)),
                "same_clip_cosine": cosine, "augmentation_top1_delta": float(delta.mean()),
                "paired_query_bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist(),
                "ci_scope": "Resample query episodes only, fixed gallery; not population/generalization CI"}
        atomic(out / "report.json", report)
        state.update(phase="complete", finished_unix=time.time())
        atomic(out / "status.json", state)
    except Exception as exc:
        state.update(phase="failed", error=repr(exc), finished_unix=time.time())
        atomic(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()

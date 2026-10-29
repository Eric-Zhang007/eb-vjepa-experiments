"""Read-only, memory-bounded EMA feature diagnostics alongside training."""
import argparse
import gc
import hashlib
import fcntl
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from app.vjepa_2_1.models.vision_transformer import vit_large

from data import decode_clip
from model import clean_state


def fixed_clips(root, output, count):
    source = root / "experiments/ctp_v1/episodes_available.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    used_episodes = {r["episode"] for r in records}
    rng = np.random.default_rng(239)
    # Exactly one clip per episode in each split.
    unique = {r["episode"]: r for r in records}
    training = list(unique.values())
    selected = [training[i] for i in rng.choice(len(training), count, replace=False)]
    camera = "observation.images.exterior_1_left"
    prefix = f"videos/{camera}/"
    fields = ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]
    heldout = []
    for parquet in sorted((root / "data/droid_1.0.1/meta/episodes").rglob("*.parquet")):
        table = pq.read_table(parquet, columns=["episode_index"] + [prefix + f for f in fields])
        for row in table.to_pylist():
            if row["episode_index"] in used_episodes:
                continue
            chunk, index, start, end = [row[prefix + f] for f in fields]
            path = root / "data/droid_1.0.1/videos" / camera / f"chunk-{chunk:03d}" / f"file-{index:03d}.mp4"
            if end - start >= 15 / 4 + 1 / 15 and path.is_file():
                heldout.append({"episode": row["episode_index"], "path": str(path),
                                "start": start, "end": end, "camera": 0})
    if len(heldout) < count:
        raise RuntimeError(f"Only {len(heldout)} eligible disjoint heldout episodes")
    test = [heldout[i] for i in rng.choice(len(heldout), count, replace=False)]
    clips = []
    for split, samples in (("train", selected), ("heldout", test)):
        for record in samples:
            clips.append({**record, "split": split,
                          "clip_start": (record["start"] + record["end"] - 15 / 4 - 1 / 15) / 2})
    (output / "fixed_clips.json").write_text(json.dumps(clips, indent=2))
    return clips


def stats(features):
    features = features.double()
    centered = features - features.mean(0)
    variance = centered.square().sum(0) / (len(features) - 1)
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square() / (len(features) - 1)
    probs = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy_rank = torch.exp(-(probs * probs.clamp_min(1e-30).log()).sum())
    normalized = F.normalize(features, dim=-1)
    similarities = normalized @ normalized.T
    pairs = similarities[torch.triu_indices(len(features), len(features), 1).unbind()]
    return {
        "samples": len(features), "max_centered_rank": min(len(features) - 1, features.shape[1]),
        "mean_dimension_std": variance.sqrt().mean().item(),
        "median_dimension_std": variance.sqrt().median().item(),
        "effective_rank_covariance_entropy": entropy_rank.item(),
        "top_eigenvalue_fraction": probs[0].item(),
        "cross_clip_cosine_mean": pairs.mean().item(),
        "cross_clip_cosine_p05": pairs.quantile(0.05).item(),
        "cross_clip_cosine_p95": pairs.quantile(0.95).item(),
        "eigenvalues_top20": eigenvalues[:20].tolist(),
    }


def full_run_clips(root, count):
    records = [json.loads(line) for line in
               (root / "experiments/ctp_full_72h/episodes.jsonl").read_text().splitlines()]
    episodes = {}
    for record in records:
        episodes.setdefault(record["episode"], {})[record["camera"]] = record
    eligible = sorted(e for e, views in episodes.items() if set(views) == {0, 1, 2})
    rng = np.random.default_rng(239)
    chosen = rng.choice(eligible, count, replace=False)
    clips = []
    for episode in chosen:
        # MP4 offsets differ by camera; use each episode-view's midpoint.
        views = episodes[int(episode)]
        for camera in range(3):
            start, end = views[camera]["start"], views[camera]["end"]
            if end - start < 15 / 4 + 1 / 15:
                raise RuntimeError("Selected episode-view is too short")
            clips.append({**views[camera], "split": f"view_{camera}",
                          "clip_start": (start + end - 15 / 4 - 1 / 15) / 2})
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--step", type=int, help="Evaluate only this EMA export on the existing fixed clips")
    parser.add_argument("--full-run", action="store_true",
                        help="Compare original and full-DROID EMA on count episodes, all three views")
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root
    output = (root / f"experiments/ctp_full_72h/diagnostics_{args.step:06d}"
              if args.full_run and args.step is not None else
              root / "experiments/ctp_v1/diagnostics_005000")
    if args.full_run and (args.step is None or args.count < 2):
        raise ValueError("Full run requires --step and count >= 2")
    output.mkdir(exist_ok=True)
    lock = (output / "diagnostics.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.full_run and (output / f"report_{args.step:06d}.json").exists():
        raise RuntimeError("Completed diagnostics exist; refusing overwrite")
    free, total = torch.cuda.mem_get_info()
    if args.min_free_gib < 4.5:
        raise ValueError("Initial headroom must be at least 4.5 GiB")
    if free < args.min_free_gib * 1024**3:
        raise RuntimeError(f"Insufficient initial free memory: {free}")
    torch.cuda.set_per_process_memory_fraction(1.5 * 1024**3 / total)
    torch.manual_seed(239)
    clips_file = output / "fixed_clips.json"
    if args.full_run:
        clips = json.loads(clips_file.read_text()) if clips_file.exists() else full_run_clips(root, args.count)
        if not clips_file.exists():
            clips_file.write_text(json.dumps(clips, indent=2))
    else:
        clips = json.loads(clips_file.read_text()) if clips_file.exists() else fixed_clips(root, output, args.count)
    sources = {
        "original": root / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt",
        "step_005000": root / "experiments/ctp_v1/embodied_vjepa_encoder_005000.pt",
    }
    if args.full_run:
        sources = {"original": sources["original"],
                   f"step_{args.step:06d}": root / f"experiments/ctp_full_72h/embodied_vjepa_encoder_{args.step:06d}.pt"}
    elif args.step is not None:
        if not clips_file.exists():
            raise RuntimeError("A follow-up requires the original fixed clip manifest")
        sources = {f"step_{args.step:06d}": root / f"experiments/ctp_v1/embodied_vjepa_encoder_{args.step:06d}.pt"}
    report = {"torch_allocator_cap_bytes": int(1.5 * 1024**3), "batch_size": 1,
              "precision": "bf16 weights and inference", "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "selection": "ema_encoder", "probe": args.probe,
              "transform": "deterministic bilinear resize short side 256, center crop 256, ImageNet normalization",
              "note": "Read-only representation diagnostics; no JEPA loss or downstream control evaluation.",
              "checkpoints": {}}
    if args.full_run:
        report["sample_scope"] = "Training-distribution diagnostic, not heldout; paired episodes across three views"
        report["fixed_clips_sha256"] = hashlib.sha256(clips_file.read_bytes()).hexdigest()
    for tag, path in sources.items():
        model = vit_large(img_size=256, num_frames=16, tubelet_size=2, patch_size=16,
                          use_rope=True, use_sdpa=True, img_temporal_dim_size=1,
                          interpolate_rope=True, n_output_distillation=1)
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(clean_state(state["ema_encoder"]), strict=True)
        del state
        model = model.eval().requires_grad_(False).to(device="cuda", dtype=torch.bfloat16)
        pooled, tokens, temporal = [], [], []
        start_time = time.monotonic()
        with torch.inference_mode():
            for i, clip in enumerate(clips[:1] if args.probe else clips):
                free, _ = torch.cuda.mem_get_info()
                if free < 2.5 * 1024**3:
                    raise RuntimeError(f"Free-memory safety floor reached: {free}; stopping diagnostics")
                array, timestamps = decode_clip(clip["path"], clip["clip_start"], clip["end"])
                x = torch.from_numpy(array).permute(0, 3, 1, 2).float()
                h, w = x.shape[-2:]
                x = F.interpolate(x, size=(round(h * 256 / min(h, w)), round(w * 256 / min(h, w))),
                                  mode="bilinear", align_corners=False)
                top, left = (x.shape[-2] - 256) // 2, (x.shape[-1] - 256) // 2
                x = x[:, :, top:top+256, left:left+256] / 255
                x = (x - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
                x = x.permute(1, 0, 2, 3).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    features = model(x).float().cpu()[0]
                torch.cuda.synchronize()
                grid = features.reshape(8, 16, 16, 1024)
                pooled.append(features.mean(0))
                tokens.append(grid[4, 8, 8])
                frame_features = grid.mean((1, 2))
                temporal.append({
                    "neighbor_cosine": F.cosine_similarity(frame_features[:-1], frame_features[1:]).mean().item(),
                    "relative_neighbor_l2": ((frame_features[1:] - frame_features[:-1]).norm(dim=-1)
                                             / frame_features[:-1].norm(dim=-1).clamp_min(1e-8)).mean().item(),
                })
                del x
                if i % 16 == 0:
                    print(json.dumps({"checkpoint": tag, "clips": i + 1,
                                      "allocated_peak": torch.cuda.max_memory_allocated(),
                                      "reserved_peak": torch.cuda.max_memory_reserved(),
                                      "free_bytes": torch.cuda.mem_get_info()[0]}), flush=True)
                time.sleep(0.15)
        pooled, tokens = torch.stack(pooled), torch.stack(tokens)
        item = {"elapsed_seconds": time.monotonic() - start_time,
                "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        if not args.probe:
            torch.save({"pooled": pooled, "center_token": tokens, "temporal": temporal}, output / f"{tag}_features.pt")
            for split in sorted({c["split"] for c in clips}):
                ids = [i for i, clip in enumerate(clips) if clip["split"] == split]
                item[split] = {
                    "pooled": stats(pooled[ids]), "center_token": stats(tokens[ids]),
                    "neighbor_cosine": float(np.mean([temporal[i]["neighbor_cosine"] for i in ids])),
                    "relative_neighbor_l2": float(np.mean([temporal[i]["relative_neighbor_l2"] for i in ids])),
                }
        report["checkpoints"][tag] = item
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if not args.probe:
        original = torch.load(output / "original_features.pt", weights_only=True)
        tag = f"step_{args.step:06d}" if args.step is not None else "step_005000"
        later = torch.load(output / f"{tag}_features.pt", weights_only=True)
        report[f"same_clip_original_{tag}_pooled_cosine_mean"] = F.cosine_similarity(
            original["pooled"], later["pooled"]).mean().item()
        if args.full_run:
            report["paired_comparison"] = {}
            for split in sorted({c["split"] for c in clips}):
                ids = [i for i, c in enumerate(clips) if c["split"] == split]
                x, y = original["pooled"][ids].double(), later["pooled"][ids].double()
                x, y = x - x.mean(0), y - y.mean(0)
                k, l = x @ x.T, y @ y.T
                report["paired_comparison"][split] = {
                    "linear_cka": ((k * l).sum() / (k.norm() * l.norm()).clamp_min(1e-30)).item(),
                    "same_clip_cosine": F.cosine_similarity(
                        original["pooled"][ids], later["pooled"][ids]).mean().item(),
                }
    name = "probe" if args.probe else "report"
    if args.step is not None:
        name += f"_{args.step:06d}"
    (output / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Prepare real LIBERO samples with frozen grid11k EMA targets. No training occurs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

TEACHER_SHA256 = "468d7af3a5bdae9fb2205718ab0f86a13f5395afcc53700500439d993baf987a"
SUITES = ("spatial", "object", "goal", "10")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episodes(root):
    """Honor declared training ranges and preserve the known Goal82 exclusion."""
    selected = []
    for suite in SUITES:
        directory = root / f"libero_{suite}_no_noops_1.0.0_lerobot"
        info = json.loads((directory / "meta/info.json").read_text())
        start, end = map(int, info["splits"]["train"].split(":"))
        for line in (directory / "meta/episodes.jsonl").read_text().splitlines():
            ep = json.loads(line)
            index = ep["episode_index"]
            if not start <= index < end or (suite == "goal" and index == 82):
                continue
            params = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
            selected.append({"suite": suite, "episode_index": index, "length": ep["length"],
                "task": ep["tasks"][0], "fps": info["fps"],
                "parquet": str(directory / info["data_path"].format(**params)),
                "video": str(directory / info["video_path"].format(**params, video_key="observation.images.image"))})
    return selected


def read_table(ep):
    import pyarrow.parquet as pq
    table = pq.read_table(ep["parquet"], columns=["action", "observation.state"])
    return {key: np.asarray(table[key].to_pylist(), dtype=np.float32) for key in table.column_names}


def fit_stats(selection):
    # Every frame in the declared training episodes, independent of --max-samples.
    rows = [read_table(ep) for ep in selection]
    out = {"method": "per_dimension_mean_std", "split": "train", "episodes": len(selection)}
    for key, name in (("action", "actions"), ("observation.state", "proprio")):
        array = np.concatenate([row[key] for row in rows]).astype(np.float64)
        out[name] = {"mean": array.mean(0).tolist(), "std": array.std(0).clip(1e-6).tolist()}
        out["frames"] = len(array)
    return out


def decode_video(path):
    import av
    with av.open(path) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/data/users/jiachangzhang/eb_vjepa"))
    p.add_argument("--data-root", type=Path, default=Path("/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=4, help="Positive smoke sample count; full cache needs parallel preparation")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.stride < 1 or args.max_samples < 1:
        p.error("stride and max-samples must be positive; this is a smoke/reference preparer")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve existing prepared data")
    teacher_path = args.root / "experiments/ctp_grid20k_20261008/embodied_vjepa_encoder_011000.pt"
    teacher_hash = sha256(teacher_path)
    if teacher_hash != TEACHER_SHA256:
        raise ValueError("Teacher does not match audited grid11k EMA export")
    selection = episodes(args.data_root)
    selection_path = args.output / "source_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2))
    print(f"Fitting statistics on {len(selection)} training episodes", flush=True)
    stats = fit_stats(selection)
    stats_path = args.output / "train_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    source = args.root / "repos/RoboActualizer/src/Roboactualizer/models/roboactualizer/vjepa.py"
    spec = importlib.util.spec_from_file_location("elaf_vjepa_factory", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    teacher = module.build_vjepa_encoder(ckpt_path=str(teacher_path), model_size="large",
        image_size=(384,384), torch_hub_dir=str(args.root / "repos/vjepa2"))
    teacher.eval().requires_grad_(False).to(args.device)
    from transformers import AutoProcessor
    from PIL import Image
    model_dir = args.root / "checkpoints/dream_vla/Dream-VLA-7B"
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=False)
    manifest = {"schema_version": 1, "smoke_only": True, "split": "train", "stats_path": "train_stats.json",
        "stats_sha256": sha256(stats_path), "source_selection": "source_selection.json",
        "source_selection_sha256": sha256(selection_path), "excluded": ["goal:82"],
        "teacher_path": str(teacher_path), "teacher_sha256": teacher_hash,
        "processor_path": str(model_dir), "processor_revision": "0cd0e25c1771c5b468c74585b5cf35c4a4aa315c",
        "target": {"frames": ["t+7", "t+8"], "fps": 20, "camera": "observation.images.image",
                   "resize": [384,384], "normalization": "ImageNet mean/std",
                   "pooling": "24x24 patch grid adaptive average to 4x4", "shape": [16,1024],
                   "latent_normalization": "per-token layer_norm over1024 dimensions, no affine, eps1e-5; no fitted statistics",
                   "boundary": "exclude samples without t+8"}, "samples": []}
    for ep in selection:
        rows = read_table(ep)
        frames = decode_video(ep["video"])
        if len(frames) != ep["length"] or len(rows["action"]) != ep["length"]:
            raise ValueError(f"Episode length mismatch: {ep['video']}")
        for t in range(0, ep["length"] - 8, args.stride):
            imgs = [Image.fromarray(frames[t]).resize((224,224))]
            conversation = [{"role":"user", "content":[{"type":"image"},
                {"type":"text", "text": f"What action should the robot take to {ep['task']}?"}]}]
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=imgs, padding=True, return_tensors="pt")
            clip = torch.from_numpy(np.stack([frames[t+7], frames[t+8]])).permute(0,3,1,2).float()/255
            clip = F.interpolate(clip, (384,384), mode="bilinear", align_corners=False, antialias=True)
            clip = (clip-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
            with torch.inference_mode():
                z = teacher(clip.permute(1,0,2,3).unsqueeze(0).to(args.device))
                if z.shape != (1,576,1024):
                    raise ValueError(f"Unexpected teacher shape {z.shape}")
                pooled = F.adaptive_avg_pool2d(z.transpose(1,2).reshape(1,1024,24,24),(4,4)).flatten(2).transpose(1,2)[0]
                target = F.layer_norm(pooled.float(), (1024,), eps=1e-5).cpu()
            sample = {k: inputs[k][0] if k in ("input_ids","attention_mask") else inputs[k]
                      for k in ("input_ids","attention_mask","pixel_values","image_grid_thw")}
            for key, source_array in (("actions",rows["action"][t:t+8]),("proprio",rows["observation.state"][t])):
                sample[key] = torch.tensor((source_array-np.array(stats[key]["mean"]))/np.array(stats[key]["std"]),dtype=torch.float32)
            provenance = {"episode_key": f"{ep['suite']}:{ep['episode_index']}", "frame_index":t,
                "future_frame_indices":[t+7,t+8], "fps":ep["fps"], "video":ep["video"], "parquet":ep["parquet"]}
            sample.update(action_valid=torch.ones(8,dtype=torch.bool), future_latents=target, provenance=provenance)
            name = f"sample_{len(manifest['samples']):08d}.pt"
            torch.save(sample,args.output/name)
            manifest["samples"].append({"path":name, "sha256":sha256(args.output/name), **provenance})
            print(f"Prepared {name} {provenance['episode_key']} frame {t}",flush=True)
            if args.max_samples and len(manifest["samples"]) >= args.max_samples:
                break
        if args.max_samples and len(manifest["samples"]) >= args.max_samples:
            break
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"COMPLETE {len(manifest['samples'])} samples",flush=True)


if __name__ == "__main__":
    main()

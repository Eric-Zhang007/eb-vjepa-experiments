"""Fixed-input 5k/10k cross-target JEPA loss; never modifies training state."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from src.masks.multiseq_multiblock3d import MaskCollator

from data import decode_clip
from model import Student, make_backbones, target_forward


def check_memory():
    free, _ = torch.cuda.mem_get_info()
    if free < 2.5 * 1024**3:
        raise RuntimeError(f"GPU free memory below safety floor: {free}")


def progress(stage, count):
    print(json.dumps({
        "stage": stage, "clips": count,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_bytes": torch.cuda.mem_get_info()[0],
    }), flush=True)


def prepare_clip(clip):
    array, _ = decode_clip(clip["path"], clip["clip_start"], clip["end"])
    x = torch.from_numpy(array).permute(0, 3, 1, 2).float()
    h, w = x.shape[-2:]
    x = F.interpolate(
        x, size=(round(h * 256 / min(h, w)), round(w * 256 / min(h, w))),
        mode="bilinear", align_corners=False,
    )
    top, left = (x.shape[-2] - 256) // 2, (x.shape[-1] - 256) // 2
    x = x[:, :, top:top + 256, left:left + 256] / 255
    x = ((x - torch.tensor([.485, .456, .406])[None, :, None, None])
         / torch.tensor([.229, .224, .225])[None, :, None, None])
    return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else None, "n": len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("This concurrent diagnostic is restricted to physical GPU7")
    torch.set_num_threads(2)
    torch.manual_seed(239)
    free, total = torch.cuda.mem_get_info()
    if free < 5 * 1024**3:
        raise RuntimeError(f"Insufficient initial free memory: {free}")
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / total)
    root = args.root
    exp = root / "experiments/ctp_v1"
    output = exp / "diagnostics_cross_005000_010000"
    output.mkdir(exist_ok=True)
    result_path = output / ("probe.json" if args.probe else "report.json")
    if result_path.exists():
        raise FileExistsError(result_path)
    manifest = exp / "diagnostics_005000/fixed_clips.json"
    clips = json.loads(manifest.read_text())
    if args.probe:
        clips = [next(c for c in clips if c["split"] == split)
                 for split in ("train", "heldout")]
    recipe = yaml.safe_load(
        (root / "repos/vjepa2/configs/train_2_1/vitl16/pretrain-256px-16f.yaml").read_text()
    )
    collator = MaskCollator(recipe["mask"], [16], crop_size=256, patch_size=16, tubelet_size=2)
    inputs, masks = [], []
    for i, clip in enumerate(clips):
        inputs.append(prepare_clip(clip))
        enc, pred = [], []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(239 + i)
            for generator in collator.mask_generators[16]:
                generator._itr_counter.value = i - 1
                me, mp = generator(1)
                enc.append(me)
                pred.append(mp)
        masks.append((enc, pred))
        if i % 32 == 0:
            progress("prepare", i + 1)
    if not args.probe:
        torch.save({"clips": clips, "masks": masks}, output / "fixed_inputs.pt")

    targets = {}
    started = time.monotonic()
    with torch.inference_mode():
        for step in (5000, 10000):
            encoder, predictor = make_backbones(False, predictor_output=1024)
            del predictor
            path = exp / f"embodied_vjepa_encoder_{step:06d}.pt"
            state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            encoder.load_state_dict(state["ema_encoder"], strict=True)
            del state
            encoder = encoder.eval().requires_grad_(False).cuda()
            targets[step] = []
            for i, x in enumerate(inputs):
                check_memory()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = target_forward(encoder, x.cuda())
                if not torch.isfinite(h).all():
                    raise FloatingPointError("Nonfinite EMA targets")
                targets[step].append(h.float().cpu())
                del h
                if i % 32 == 0:
                    progress(f"target_{step}", i + 1)
                time.sleep(.15)
            del encoder
            gc.collect()
            torch.cuda.empty_cache()

        losses = {}
        for step in (5000, 10000):
            encoder, predictor = make_backbones(False, predictor_output=1024)
            student = Student(encoder, predictor)
            del encoder, predictor
            state = torch.load(exp / f"step_{step:06d}.pt", map_location="cpu",
                               weights_only=False, mmap=True)
            if state["step"] != step:
                raise RuntimeError("Checkpoint step mismatch")
            student.encoder.load_state_dict(state["encoder"], strict=True)
            student.predictor.load_state_dict(state["predictor"], strict=True)
            del state
            student = student.eval().requires_grad_(False).cuda()
            if student.context_schedule.value(step - 1) != 0:
                raise RuntimeError("Expected zero context coefficient for these checkpoints")
            for target_step in (5000, 10000):
                losses[f"student_{step}_target_{target_step}"] = []
            for i, x in enumerate(inputs):
                check_memory()
                enc, pred = [[m.cuda() for m in group] for group in masks[i]]
                x_gpu = x.cuda()
                for target_step in (5000, 10000):
                    h = targets[target_step][i].cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, components = student(x_gpu, h, enc, pred, step - 1)
                    if not torch.isfinite(loss) or not torch.isfinite(components).all():
                        raise FloatingPointError("Nonfinite diagnostic loss")
                    losses[f"student_{step}_target_{target_step}"].append(
                        {"loss": loss.item(), "components": components.cpu().tolist()}
                    )
                    del h, loss, components
                del x_gpu, enc, pred
                if i % 32 == 0:
                    progress(f"student_{step}", i + 1)
                time.sleep(.15)
            del student
            gc.collect()
            torch.cuda.empty_cache()

    report = {
        "steps": [5000, 10000], "gpu": 7, "probe": args.probe,
        "precision": "FP32 weights, BF16 autocast, matching trainer",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "mask_seed": 239, "mask_recipe": recipe["mask"],
        "transform": "deterministic short-side 256 bilinear resize, center crop, ImageNet normalization",
        "objective": "trainer Student.forward; prediction L1, context coefficient zero",
        "target": "ema_encoder, trainer target_forward including layer_norm",
        "limitations": "One mask draw per clip, batch 1, deterministic crops; not comparable in absolute value to augmented training loss. Heldout episodes are not guaranteed site/task disjoint.",
        "elapsed_inference_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "splits": {},
    }
    for split in ("train", "heldout"):
        ids = [i for i, clip in enumerate(clips) if clip["split"] == split]
        cells = {}
        vectors = {}
        for key, records in losses.items():
            vector = np.asarray([records[i]["loss"] for i in ids])
            vectors[key] = vector
            cells[key] = {"loss": summary(vector), "mask_prediction": [
                summary([records[i]["components"][mask][0] for i in ids])
                for mask in range(2)]}
        a = vectors["student_5000_target_5000"]
        b = vectors["student_5000_target_10000"]
        c = vectors["student_10000_target_5000"]
        d = vectors["student_10000_target_10000"]
        report["splits"][split] = {
            "cells": cells,
            "paired_deltas": {
                "student_change_fixed_target_5000": summary(c - a),
                "student_change_fixed_target_10000": summary(d - b),
                "target_change_fixed_student_5000": summary(b - a),
                "target_change_fixed_student_10000": summary(d - c),
                "diagonal_change": summary(d - a),
            },
            "normalized_target_l1_drift": summary([
                (targets[5000][i] - targets[10000][i]).abs().mean().item() for i in ids]),
        }
    result_path.write_text(json.dumps(report, indent=2))
    if not args.probe:
        (output / "per_clip.json").write_text(json.dumps({"clips": clips, "losses": losses}))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

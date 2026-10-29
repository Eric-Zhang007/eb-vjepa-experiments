"""Metadata alignment and optional RANDOM-WEIGHT GPU architecture unit test.

This does not claim pretrained initialization, calibration, or training smoke.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from src.masks.multiseq_multiblock3d import MaskCollator

from model import Student, clean_state, make_backbones, target_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(239)
    metadata = json.loads((args.root / "experiments/ctp_v1/checkpoint_metadata_only.json").read_text())
    # Upstream builds drop-path scalars with Tensor.item(), incompatible with meta.
    encoder, predictor = make_backbones(True)
    result = {}
    for name, net, source in (
        ("encoder", encoder, "ema_encoder"), ("predictor", predictor, "predictor"),
    ):
        state = clean_state(metadata["subtrees"][source])
        expected = net.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatch = {k: [list(v.shape), state[k]["shape"]]
                    for k, v in expected.items()
                    if k in state and list(v.shape) != state[k]["shape"]}
        result[name] = {"source": source, "missing": missing, "unexpected": unexpected,
                        "mismatched_shapes": mismatch}
        assert not (missing or unexpected or mismatch), result[name]
    del encoder, predictor
    if args.gpu:
        print("RANDOM-WEIGHT architecture test, NOT pretrained smoke", flush=True)
        encoder, predictor = make_backbones(True, predictor_output=1024)
        target = copy.deepcopy(encoder).requires_grad_(False).cuda().eval()
        target.use_activation_checkpointing = False
        student = Student(encoder, predictor).cuda().train()
        recipe = yaml.safe_load((args.root / "repos/vjepa2/configs/train_2_1/vitl16/pretrain-256px-16f.yaml").read_text())
        collator = MaskCollator(recipe["mask"], [16], crop_size=256, patch_size=16, tubelet_size=2)
        masks = [generator(1) for generator in collator.mask_generators[16]]
        enc = [pair[0].cuda() for pair in masks]
        pred = [pair[1].cuda() for pair in masks]
        clips = torch.randn(1, 3, 16, 256, 256, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = target_forward(target, clips)
            loss, components = student(clips, h, enc, pred, 16000)
        loss.backward()
        assert torch.isfinite(loss)
        gradient = student.encoder.patch_embed.proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0
        assert all(p.grad is None for p in target.parameters())
        result["random_weight_gpu_unit"] = {
            "passed": True, "loss": loss.item(), "components": components.tolist(),
            "encoder_gradient_norm": gradient.norm().item(),
            "target_shape": list(h.shape),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
    result["full_pretrained_load_verified"] = False
    (args.root / "experiments/ctp_v1/model_unit_test.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

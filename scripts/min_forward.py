#!/usr/bin/env python3
"""Minimal V-JEPA forward smoke test (ViT-L/16, 256px, 16 frames, bf16)."""
import sys

import torch

REPO = "/data/users/jiachangzhang/eb_vjepa/repos/vjepa2"


def main():
    sys.path.insert(0, REPO)
    from src.models import vision_transformer as video_vit

    model = video_vit.vit_large(
        img_size=256,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        use_sdpa=True,
    )
    model = model.to("cuda").eval()
    n = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 16, 256, 256, device="cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    print("params_M=%.1f" % (n / 1e6), "in=", tuple(x.shape), "out=", tuple(y.shape), "dtype=", y.dtype)


if __name__ == "__main__":
    main()

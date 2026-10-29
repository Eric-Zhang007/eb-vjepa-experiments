"""CPU-only real LIBERO video and strict encoder checkpoint integration check."""
import argparse
import gc
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from Roboactualizer.models.roboactualizer.backbone import VJEPABackbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--step', type=int, default=20000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    table = pq.read_table(args.dataset / 'data/chunk-000/episode_000000.parquet')
    records = table.slice(0, 7).to_pylist()
    indices = [0, 2, 4, 6]
    cameras = []
    for camera in ('image', 'wrist_image'):
        path = args.dataset / f'videos/chunk-000/observation.images.{camera}/episode_000000.mp4'
        selected = []
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in indices:
                    selected.append(frame.to_ndarray(format='rgb24'))
                if index >= indices[-1]:
                    break
        if len(selected) != 4:
            raise RuntimeError('Incomplete causal observation clip')
        x = torch.from_numpy(np.stack(selected)).permute(0, 3, 1, 2).float()
        cameras.append(F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False))
    clip = (torch.cat(cameras, dim=-1) / 127.5 - 1).unsqueeze(0)
    report = {'dataset': str(args.dataset), 'frame_indices': indices,
              'clip_shape': list(clip.shape), 'device': 'cpu',
              'action_shape': list(np.asarray(records[-1]['action']).shape),
              'state_shape': list(np.asarray(records[-1]['observation.state']).shape),
              'task_index': records[-1]['task_index'],
              'limitations': 'Shape/loading test only; not full dataset adapter, text encoder or policy training.',
              'encoders': {}}
    sources = {'original': args.root / 'checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt',
               'embodied': args.root / f'experiments/ctp_v1/embodied_vjepa_encoder_{args.step:06d}.pt'}
    for name, checkpoint in sources.items():
        model = VJEPABackbone(ckpt_path=str(checkpoint),
                             torch_hub_dir=str(args.root / 'repos/vjepa2'),
                             image_size=(224, 448), device='cpu')
        with torch.inference_mode():
            z = model.encode_clip_last_group(clip)
        assert z.shape == (1, 392, 1024) and torch.isfinite(z).all()
        assert not any(p.requires_grad for p in model.parameters())
        report['encoders'][name] = {'checkpoint': str(checkpoint), 'strict_load': True,
                                    'shape': list(z.shape), 'std': z.std().item()}
        print(json.dumps(report['encoders'][name]), flush=True)
        del model, z
        gc.collect()
    output = args.root / 'experiments/action_only'
    output.mkdir(exist_ok=True)
    (output / 'real_encoder_check.json').write_text(json.dumps(report, indent=2))
    print('PASS real-video encoder integration', flush=True)


if __name__ == '__main__':
    main()

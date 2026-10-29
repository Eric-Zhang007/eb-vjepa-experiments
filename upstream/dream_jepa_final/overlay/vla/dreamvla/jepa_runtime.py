"""Frozen grid11k runtime, isolated from optimizer and checkpoint parameters."""
import importlib.util
from pathlib import Path
import torch
import json
import numpy as np
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from torch.nn import functional as F


class FrozenGrid11k:
    def __init__(self, root, checkpoint, device, batch_size=4):
        root = Path(root)
        source = root / 'repos/RoboActualizer/src/Roboactualizer/models/roboactualizer/vjepa.py'
        spec = importlib.util.spec_from_file_location('dream_grid11k_factory', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.encoder = module.build_vjepa_encoder(ckpt_path=checkpoint, model_size='large',
            image_size=(384, 384), torch_hub_dir=str(root / 'repos/vjepa2')).eval().requires_grad_(False).to(device)
        self.device, self.batch_size = device, batch_size
        self.cache_root = root / 'data/elaf_v0_grid11k_raw'
        self.index = json.loads((root / 'experiments/dream_jepa_dualview_20261023/episode_index.json').read_text())
        self.maps = OrderedDict()
        self.cache = None
        self.cache_namespace = None
        self.cache_write = None
        self.cache_writer = None
        if os.environ.get('DREAM_JEPA_RAM_SOCKET'):
            from dreamvla.jepa_cache import Client
            self.cache = Client(os.environ['DREAM_JEPA_RAM_SOCKET'])
            self.cache_put_client = Client(os.environ['DREAM_JEPA_RAM_SOCKET'])
            self.cache_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jepa-cache-put')
            digest = hashlib.sha256()
            with open(checkpoint, 'rb') as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                    digest.update(chunk)
            self.cache_namespace = digest.hexdigest() + ':wrist4_384_bilinear_aa_imagenet_bf16_compute_fp32_output_v2:'
            self.cache.request('stats')
        self.last_metrics = {}
        # Inspect the actual encoder rather than trusting token count alone.
        enc = self.encoder
        convs = [m for m in enc.modules() if isinstance(m, torch.nn.Conv3d)]
        if not any(m.kernel_size == (2, 16, 16) and m.stride == (2, 16, 16) for m in convs):
            raise ValueError('grid11k requires verified (2,16,16) patch embedding')

    @torch.no_grad()
    def __call__(self, batch):
        started = time.monotonic()
        primary, future, wrist, keys = [], [], [], []
        for key, step, name in zip(batch['jepa_episode_key'], batch['jepa_frame'], batch['dataset_names']):
            name = name.decode() if isinstance(name, bytes) else str(name)
            identity = name + ':' + bytes(key.tolist()).hex()
            ep = self.index[identity]
            if identity not in self.maps:
                directory = self.cache_root / ep['path']
                self.maps[identity] = (np.load(directory/'features.npy', mmap_mode='r'),
                                       np.load(directory/'wrist_images.npy', mmap_mode='r'))
                while len(self.maps) > 32:
                    self.maps.popitem(last=False)
            features, images = self.maps[identity]
            self.maps.move_to_end(identity)
            t = int(step)
            keys.append((self.cache_namespace or '') + identity + ':' + str(t))
            primary.append(torch.from_numpy(features[t].copy()).view(torch.bfloat16))
            future.append(torch.from_numpy(features[min(t+8,len(features)-1)].copy()).view(torch.bfloat16))
            wrist.append(np.asarray(images[np.maximum(np.arange(t-3,t+1),0)]))
        ready = time.monotonic()
        cached = [None]*len(keys) if self.cache is None else self.cache.get(keys)
        missing = [i for i, value in enumerate(cached) if value is None]
        outputs = [None]*len(keys)
        for i, value in enumerate(cached):
            if value is not None:
                outputs[i] = torch.from_numpy(np.frombuffer(value, dtype=np.float32).copy()).reshape(1152,1024).to(self.device)
        cache_done = time.monotonic()
        new_pairs = []
        if missing:
            videos = torch.from_numpy(np.stack([wrist[i] for i in missing])).pin_memory()
            encoded = self.encode_videos(videos, pad_batch=self.cache is not None)
            for j, i in enumerate(missing):
                outputs[i] = encoded[j]
            if self.cache is not None:
                if encoded.dtype != torch.float32:
                    raise ValueError(f'Unexpected frozen encoder output dtype: {encoded.dtype}')
                if self.cache_write is not None:
                    self.cache_write.result()  # Bound outstanding copies; surface write failures.
                host = torch.empty(encoded.shape, dtype=encoded.dtype, device='cpu', pin_memory=True)
                host.copy_(encoded, non_blocking=True)
                copied = torch.cuda.Event()
                copied.record()
                def publish(host, event, selected_keys):
                    event.synchronize()
                    array = host.numpy()
                    return self.cache_put_client.put([(key, array[j].tobytes()) for j, key in enumerate(selected_keys)])
                self.cache_write = self.cache_writer.submit(publish, host, copied, [keys[i] for i in missing])
        self.last_metrics = dict(jepa_cache_hits=len(keys)-len(missing), jepa_cache_requests=len(keys),
                                 jepa_cpu_read_seconds=ready-started, jepa_cache_get_seconds=cache_done-ready,
                                 jepa_prepare_seconds=time.monotonic()-started)
        return (torch.stack(primary).pin_memory().to(self.device, non_blocking=True), torch.stack(outputs),
                torch.stack(future).pin_memory().to(self.device, non_blocking=True))

    @torch.no_grad()
    def encode_videos(self, videos, pad_batch=False):
        outputs = []
        for clip in videos.split(self.batch_size):
            count = len(clip)
            if pad_batch and count < self.batch_size:
                clip = torch.cat([clip, clip[-1:].expand(self.batch_size-count, *clip.shape[1:])])
            clip = clip.to(self.device, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255
            b, t, c, h, w = clip.shape
            if t != 4:
                raise ValueError('JEPA requires four consecutive same-view frames')
            clip = F.interpolate(clip.reshape(b*t, c, h, w), (384,384), mode='bilinear', align_corners=False, antialias=True)
            mean = clip.new_tensor([.485,.456,.406])[None,:,None,None]
            std = clip.new_tensor([.229,.224,.225])[None,:,None,None]
            clip = ((clip-mean)/std).reshape(b,t,c,384,384).permute(0,2,1,3,4)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = self.encoder(clip)
            if features.shape != (b,1152,1024):
                raise ValueError(f'Unexpected raw encoder grid: {features.shape}')
            outputs.append(features[:count])
        return torch.cat(outputs)

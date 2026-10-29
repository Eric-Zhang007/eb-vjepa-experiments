"""Paired checkpoint diagnostics, separate Action/Delta backward; no optimizer steps."""
import argparse
import json
import re
from pathlib import Path
import torch
from elaf.v0_train import build_backbone, model_config
from elaf.v0_model import ELAFV0Policy
from elaf.v0_dataset import CachedV0Dataset, collate_v0
from elaf.v0_training import seed_all, load_trainable_state, atomic_json


def stats(x):
    x = x.detach().float()
    return {'rms': x.square().mean().sqrt().item(), 'max_abs': x.abs().max().item()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    config = json.loads((args.run_dir/'resolved_config.json').read_text())['config']
    seed_all(731)
    policy = ELAFV0Policy(build_backbone(config), model_config(config)).cuda().eval()
    dataset = CachedV0Dataset(config['manifest'], config['checkpoint'], image_aug=False)
    indices = []
    for suite in sorted(dataset.suite_indices):
        indices.append(next(i for i in dataset.suite_indices[suite] if bool(dataset[i]['future_valid'])))
    lora = [(n, p) for n, p in policy.named_parameters() if 'lora_' in n and p.requires_grad]
    def layer(n):
        match = re.search(r'\.layers\.(\d+)\.', n)
        return int(match.group(1)) if match else -1
    recording = {'active': False, 'activations': {}, 'adapters': {}}
    def block_hook(name):
        def hook(module, inputs, output):
            if not recording['active']: return
            h = output[0] if isinstance(output, tuple) else output
            recording['activations'][name] = {'all': stats(h), 'current': stats(h[:, -72:-40]),
                'delta': stats(h[:, -40:-8]), 'action': stats(h[:, -8:])}
        return hook
    def adapter_hook(name):
        def hook(module, inputs, output):
            if not recording['active']: return
            with torch.no_grad():
                x = inputs[0].detach()
                base = module.base_layer(x)
                delta = torch.zeros_like(base, dtype=torch.float32)
                for adapter in module.active_adapters:
                    if adapter in module.lora_A:
                        z = x.to(module.lora_A[adapter].weight.dtype)
                        delta += (module.lora_B[adapter](module.lora_A[adapter](module.lora_dropout[adapter](z))) * module.scaling[adapter]).float()
                b, d = stats(base), stats(delta)
                recording['adapters'][name] = {'layer': layer(name), 'base': b, 'lora': d,
                    'ratio': d['rms']/max(b['rms'], 1e-30)}
        return hook
    handles = []
    for name, module in policy.named_modules():
        if re.search(r'\.layers\.\d+$', name): handles.append(module.register_forward_hook(block_hook(name)))
        if hasattr(module, 'lora_A') and hasattr(module, 'base_layer'):
            handles.append(module.register_forward_hook(adapter_hook(name)))
    records = []
    checkpoints = [args.run_dir/'preserved_step3000', args.run_dir/'preserved_latest_paused']
    for checkpoint in checkpoints:
        step = json.loads((checkpoint/'progress.json').read_text())['progress']['step']
        state = torch.load(checkpoint/'training.pt', map_location='cpu', weights_only=False)
        load_trainable_state(policy, state['model']); del state
        for index in indices:
            batch = collate_v0([dataset[index]], pad_token_id=dataset.pad_token_id)
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            for time in [.25, .5, .75]:
                seed_all(81000 + index)
                nd = torch.randn((1, 32, 1024), device='cuda', dtype=torch.float32)
                na = torch.randn((1, 8, 7), device='cuda', dtype=torch.bfloat16)
                t = torch.full((1,), time, device='cuda')
                recording.update(active=True, activations={}, adapters={})
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses = policy.loss(batch, u=t, s=t, delta_noise=nd, action_noise=na, return_debug=True)
                recording['active'] = False
                ga = torch.autograd.grad(losses['action_loss'], [p for _, p in lora], retain_graph=True, allow_unused=True)
                ga_cpu = [g.detach().float().cpu() if g is not None else None for g in ga]; del ga
                gd = torch.autograd.grad(losses['delta_loss'], [p for _, p in lora], allow_unused=True)
                sums = {}
                for (name, p), a, d in zip(lora, ga_cpu, gd):
                    k = layer(name); row = sums.setdefault(k, [0., 0., 0.])
                    if a is not None: row[0] += a.double().square().sum().item()
                    if d is not None:
                        d = d.detach().float().cpu().double(); row[1] += d.square().sum().item()
                        if a is not None: row[2] += (a.double()*d).sum().item()
                def summarize(v):
                    a, d, dot = v
                    return {'action_grad_norm': a**.5, 'delta_grad_norm': d**.5,
                        'cosine': dot/(a*d)**.5 if a*d > 0 else None}
                record = {'checkpoint': str(checkpoint), 'step': step, 'sample_id': batch['sample_id'],
                    'dataset_index': index, 'u': time, 's': time,
                    'action_loss': losses['action_loss'].item(), 'delta_loss': losses['delta_loss'].item(),
                    'layers': {str(k): summarize(v) for k, v in sums.items()},
                    'global': summarize([sum(v[j] for v in sums.values()) for j in range(3)]),
                    'activations': recording['activations'], 'adapters': recording['adapters']}
                records.append(record)
                atomic_json(args.output/'results.json', {'protocol': '4 training samples, one per suite; no augmentation; eval mode; paired seeds; u=s=.25,.5,.75; no optimizer updates; original BF16 backbone and FP32 resampler; separate unscaled unclipped loss gradients', 'records': records})
                print('completed', step, index, time, record['global'], flush=True)
                del gd, ga_cpu, losses
    for handle in handles: handle.remove()
    atomic_json(args.output/'status.json', {'status': 'completed', 'records': len(records)})


if __name__ == '__main__': main()

"""V2 persistence and sustained-instability guard, reusing v0 checkpoint I/O."""
import hashlib
import math
import os
from pathlib import Path
import shutil
import statistics
import torch
import torch.distributed as dist
from elaf.v0_training import save_checkpoint, barrier, atomic_json


def lora_digest(policy):
    h = hashlib.sha256()
    for name, p in policy.named_parameters():
        if 'lora_' in name:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_v2(directory, policy, optimizer, scheduler, progress, contract, rank):
    # v0 save selects requires_grad parameters. Include frozen learned LoRA too.
    frozen = [p for n, p in policy.named_parameters() if 'lora_' in n and not p.requires_grad]
    for p in frozen: p.requires_grad_(True)
    try:
        final = save_checkpoint(directory, policy, optimizer, scheduler, progress, contract, rank)
    finally:
        for p in frozen: p.requires_grad_(False)
    if rank == 0:
        run = Path(directory).parent
        step = progress['step']
        if step in contract['config']['preserve_steps']:
            dest = run/'preserved'/f'step-{step:06d}'
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True)
                shutil.copytree(final, dest, copy_function=os.link)
            if step in contract['config']['full_diagnostic_steps']:
                queue = run/'full_diagnostic_queue'; queue.mkdir(exist_ok=True)
                atomic_json(queue/f'step-{step:06d}.json', {'step': step, 'checkpoint': str(dest.resolve())})
    barrier()
    return final


class StabilityGuard:
    """Non-overlapping windows; pause only for sustained gradient+loss worsening."""
    def __init__(self, config, state=None):
        self.config = config
        self.state = state or {'window': [], 'baseline': None, 'bad_windows': 0, 'history': []}

    def update(self, record):
        c, s = self.config, self.state
        values = [record[k] for k in ['train/grad/global_preclip', 'train/action_loss', 'train/delta_loss']]
        if not all(math.isfinite(x) for x in values): return True, 'nonfinite_metrics'
        s['window'].append(values)
        if len(s['window']) < c['window']: return False, None
        med = [statistics.median(x[i] for x in s['window']) for i in range(3)]
        s['window'] = []
        s['history'].append({'step': record['step'], 'medians': med})
        if record['step'] < c['baseline_start']: return False, None
        if s['baseline'] is None:
            s['baseline'] = med; return False, None
        b = s['baseline']
        grad_bad = med[0] > max(c['gradient_absolute_floor'], c['gradient_ratio'] * b[0])
        loss_bad = any(med[i] > max(b[i]*c['loss_ratio'], b[i]+c['loss_absolute_increase']) for i in [1,2])
        s['bad_windows'] = s['bad_windows'] + 1 if grad_bad and loss_bad else 0
        return s['bad_windows'] >= c['consecutive_windows'], 'sustained_gradient_and_loss_growth'


def action_only_backward(policy, action_loss, total_loss):
    """Exact partial-gradient routing; non-LoRA receives both losses unchanged."""
    named = [(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    lora = [p for n,p in named if 'lora_' in n]
    other = [p for n,p in named if 'lora_' not in n]
    groups = []
    if lora: groups.append((lora, torch.autograd.grad(action_loss, lora, retain_graph=True, allow_unused=True)))
    groups.append((other, torch.autograd.grad(total_loss, other, allow_unused=True)))
    for params, grads in groups:
        for p,g in zip(params, grads):
            value = torch.zeros_like(p) if g is None else g.detach()
            if dist.is_initialized():
                dist.all_reduce(value); value /= dist.get_world_size()
            if p.grad is None: p.grad = value
            else: p.grad.add_(value)

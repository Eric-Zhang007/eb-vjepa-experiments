"""Diagnostic fork: full optimizer/RNG/cursor resume, only LoRA LR becomes zero."""
import hashlib
import torch
import elaf.v0_train as train
from elaf.v0_training import atomic_json
from pathlib import Path
import sys

original_load = train.load_checkpoint


def resume_control(path, policy, optimizer, scheduler, contract, rank):
    progress = original_load(path, policy, optimizer, scheduler, contract, rank)
    assert progress['step'] == 3000
    for i, group in enumerate(optimizer.param_groups):
        if group['group_name'] == 'lora':
            group['lr'] = 0.0
            group['initial_lr'] = 0.0
            scheduler.base_lrs[i] = 0.0
            scheduler._last_lr[i] = 0.0
    # New W&B identity and selection window; preserve absolute step/data/RNG.
    progress.update(wandb_run_id=None, best_loss=None, window_loss=0., window_count=0, is_best=False)
    if rank == 0:
        output = Path(sys.argv[sys.argv.index('--output')+1])
        h = hashlib.sha256()
        for name, parameter in policy.named_parameters():
            if 'lora_' in name:
                h.update(name.encode());h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        atomic_json(output/'control_receipt.json', {'parent_step': 3000, 'parent_checkpoint': str(path),
            'intervention': 'LoRA lr and scheduler base_lr zero; requires_grad retained; all optimizer moments retained',
            'initial_lora_sha256': h.hexdigest(), 'data_rng_cursor': 'restored', 'warmup': 'not restarted'})
    return progress


if __name__ == '__main__':
    train.load_checkpoint = resume_control
    train.main()

"""Prepare an explicit LR-only continuation without resetting training state."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import torch
import numpy as np


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--factor', type=float, default=10.)
    parser.add_argument('--decision-step', type=int)
    args = parser.parse_args()
    assert args.factor > 0
    status = json.loads((args.parent/'status.json').read_text())
    assert status['status'] in ('paused', 'decision_required'), status
    source = (args.parent/'checkpoints/latest').resolve()
    metadata = json.loads((source/'progress.json').read_text())
    step = metadata['progress']['step']
    assert status['step'] == step and metadata['progress']['stage'] == 3
    args.output.mkdir(parents=True, exist_ok=False)
    preserved = args.parent/'preserved'/f'lr-fork-step-{step:06d}'
    if not preserved.exists():
        shutil.copytree(source, preserved, copy_function=os.link)
    destination = args.output/'fork_checkpoint'
    destination.mkdir()
    state = torch.load(preserved/'training.pt', map_location='cpu', weights_only=False)
    old_contract = copy.deepcopy(metadata['contract'])
    config = metadata['contract']['config']
    for section in ('stage3', 'stage3_min'):
        config['stages'][section]['lora_lr'] *= args.factor
    if args.decision_step is not None:
        assert step < args.decision_step <= config['max_steps']
        config['decision_step'] = args.decision_step
        config['preserve_steps'] = sorted(set(config['preserve_steps'] + [args.decision_step]))
        config['full_diagnostic_steps'] = sorted(set(config['full_diagnostic_steps'] + [args.decision_step]))
    state['scheduler']['config'] = copy.deepcopy(config['stages'])
    parent_id = metadata['progress']['wandb_run_id']
    metadata['progress']['wandb_run_id'] = None
    # Optimizer tensors and group settings are retained verbatim. StageSchedule
    # applies the new LR before the first resumed update, at the original step.
    torch.save(state, destination/'training.pt')
    (destination/'progress.json').write_text(json.dumps(metadata, indent=2))
    for rank in range(metadata['contract']['world_size']):
        shutil.copy2(preserved/f'rank_{rank}.pt', destination/f'rank_{rank}.pt')
    restored = torch.load(destination/'training.pt', map_location='cpu', weights_only=False)
    for field in ('model', 'optimizer'):
        equal(state[field], restored[field])
    for rank in range(metadata['contract']['world_size']):
        equal(torch.load(preserved/f'rank_{rank}.pt', weights_only=False),
              torch.load(destination/f'rank_{rank}.pt', weights_only=False))
    for field in ('stage', 'completed_step', 'total_steps'):
        assert restored['scheduler'][field] == state['scheduler'][field]
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    receipt = dict(parent_run_id=parent_id, source=str(preserved), step=step,
                   factor=args.factor, old_contract=old_contract,
                   new_contract=metadata['contract'],
                   verification='model, optimizer and six rank RNG/cursor states retained; scheduler stage and step retained',
                   schedule='multiply Stage 3 LoRA peak and floor; retain global schedule; no new warmup',
                   decision_step=config['decision_step'])
    (args.output/'fork_receipt.json').write_text(json.dumps(receipt, indent=2))
    diag = args.output/'v2_diagnostics'
    diag.mkdir()
    shutil.copy2(args.parent/'v2_diagnostics/representation-000000000.json',
                 diag/'representation-000000000.json')
    print(json.dumps({k:v for k,v in receipt.items() if 'contract' not in k}))


if __name__ == '__main__':
    main()

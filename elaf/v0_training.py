"""E-LAF v0 deterministic scheduling and complete, atomic training state."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
import torch.distributed as dist


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def sample_indices(suite_indices, *, seed, micro_step, rank, world_size, batch_size):
    """Equal suite probability, then uniform frame; no global RNG consumption."""
    suites = sorted(suite_indices)
    if len(suites) != 4 or any(not len(suite_indices[s]) for s in suites):
        raise ValueError('Expected four nonempty LIBERO suites')
    generator = torch.Generator().manual_seed(seed + micro_step)
    selected = []
    for _ in range(world_size * batch_size):
        suite = suites[torch.randint(4, (), generator=generator).item()]
        members = suite_indices[suite]
        selected.append(members[torch.randint(len(members), (), generator=generator).item()])
    return selected[rank * batch_size:(rank + 1) * batch_size]


def parameter_groups(policy, config):
    """Separate clipping domains without changing learning-rate ownership."""
    partition = config.get('clip_partition', 'default')
    if partition not in ('default', 'flow'):
        raise ValueError(f'Unknown clip_partition: {partition}')
    flow_modules = {
        'action_in': 'action', 'action_out': 'action',
        'action_time': 'action', 'action_position': 'action',
        'delta_in': 'delta', 'delta_out': 'delta',
        'delta_time': 'delta', 'delta_position': 'delta',
        'current_in': 'current', 'current_position': 'current',
        'proprio_in': 'current',
    }
    buckets = {}
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = 'lora' if name.startswith('backbone.') else 'resampler' if name.startswith('resampler.') else 'new_modules'
        if partition == 'flow' and kind == 'new_modules':
            module = name.split('.', 1)[0]
            if module not in flow_modules:
                raise ValueError(f'Unassigned trainable parameter in flow partition: {name}')
            kind = flow_modules[module]
        decay = config['weight_decay'] if parameter.ndim >= 2 and not any(
            tag in name.lower() for tag in ('norm', 'position', 'queries')) else 0.0
        buckets.setdefault((kind, decay), []).append(parameter)
    return [dict(params=params, lr=config[(kind if kind in ('lora', 'resampler') else 'new_modules') + '_lr'], weight_decay=decay, group_name=kind)
            for (kind, decay), params in buckets.items()]


def cosine_multiplier(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1., (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(optimizer, config):
    """Schedule optimizer updates, matching the selected experiment contract."""
    options = config['optimizer']
    if options['scheduler'] == 'warmup_constant':
        warmup = int(options['warmup_steps'])
        if warmup < 1:
            raise ValueError('Recovery warmup_steps must be positive')
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: min(1.0, (step + 1) / warmup))
    if options['scheduler'] == 'multistep':
        if options.get('warmup_ratio', 0) != 0:
            raise ValueError('Baseline MultiStepLR requires zero warmup')
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=options['milestones'], gamma=options['gamma'])
    if options['scheduler'] == 'cosine':
        warmup = round(config['max_steps'] * options['warmup_ratio'])
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: cosine_multiplier(step, config['max_steps'], warmup))
    raise ValueError(f"Unknown scheduler: {options['scheduler']}")


def clip_grad_groups(optimizer, max_norm=1.0):
    """Clip logical groups independently, combining their decay/no-decay buckets."""
    if max_norm <= 0:
        raise ValueError('max_norm must be positive')
    groups = {}
    seen = set()
    for group in optimizer.param_groups:
        name = group['group_name']
        for parameter in group['params']:
            if id(parameter) in seen:
                raise ValueError('Parameter occurs in multiple optimizer groups')
            seen.add(id(parameter))
            if parameter.grad is not None:
                groups.setdefault(name, []).append(parameter)
    norms = {}
    for name, parameters in groups.items():
        pre = torch.nn.utils.clip_grad_norm_(parameters, max_norm,
                                             error_if_nonfinite=True)
        post = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters]))
        norms[name] = dict(preclip=float(pre), postclip=float(post))
    return dict(global_preclip=math.sqrt(sum(v['preclip'] ** 2 for v in norms.values())),
                groups=norms)


def trainable_state(policy):
    names = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    names.update(name for name, _ in policy.named_buffers())
    return {name: tensor.detach().cpu().clone() for name, tensor in policy.state_dict().items() if name in names}


def load_trainable_state(policy, saved):
    expected = trainable_state(policy)
    if set(expected) != set(saved):
        raise ValueError('Checkpoint trainable parameter/buffer names differ')
    for key in expected:
        if expected[key].shape != saved[key].shape:
            raise ValueError(f'Checkpoint shape mismatch: {key}')
    policy.load_state_dict(saved, strict=False)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(directory, policy, optimizer, scheduler, progress, contract, rank):
    """Every rank saves RNG/cursor; only complete generations become visible."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    token = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(token, src=0)
    temporary = directory / ('.generation_' + token[0])
    final = directory / f"generation_{progress['step']:06d}_{token[0]}"
    if rank == 0:
        temporary.mkdir()
    barrier()
    local_rng = rng_state()
    torch.save(dict(rng=local_rng, micro_step=progress['micro_step']), temporary / f'rank_{rank}.pt')
    if rank == 0:
        torch.save(dict(model=trainable_state(policy), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict()), temporary / 'training.pt')
        atomic_json(temporary / 'progress.json', dict(progress=progress, contract=contract))
    barrier()
    if rank == 0:
        os.replace(temporary, final)
        for name in ('latest', 'best'):
            if name == 'best' and not progress['is_best']:
                continue
            link = directory / ('.' + name + '.tmp')
            if link.is_symlink():
                link.unlink()
            link.symlink_to(final.name, target_is_directory=True)
            os.replace(link, directory / name)
        keep = {p.resolve() for p in (directory / 'latest', directory / 'best') if p.exists()}
        for old in directory.glob('generation_*'):
            if old.is_dir() and old.resolve() not in keep:
                shutil.rmtree(old)
    barrier()
    restore_rng(local_rng)
    return final


def load_checkpoint(path, policy, optimizer, scheduler, contract, rank):
    path = Path(path)
    metadata = json.loads((path / 'progress.json').read_text())
    if metadata['contract'] != contract:
        raise ValueError('Resume contract mismatch: config, source, data and world size must match')
    state = torch.load(path / 'training.pt', map_location='cpu', weights_only=False)
    load_trainable_state(policy, state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    local = torch.load(path / f'rank_{rank}.pt', map_location='cpu', weights_only=False)
    if local['micro_step'] != metadata['progress']['micro_step']:
        raise ValueError('Rank cursor differs from checkpoint progress')
    restore_rng(local['rng'])
    return metadata['progress']


def load_fork_checkpoint(path, policy, optimizer, rank, reset_optimizer=True, *, contract=None):
    """Initialize a NEW run; retain rank RNG/cursor, never restore scheduler.

    The caller must retain returned micro_step, but reset step, best-loss and W&B
    identity. Pass the new contract to validate frozen assets, data and topology;
    optimization/source/precision changes are deliberately allowed for recovery.
    If moments are retained, the NEW optimizer's hyperparameters remain in force.
    """
    path = Path(path)
    metadata = json.loads((path / 'progress.json').read_text())
    if contract is not None:
        old = metadata['contract']
        for key in ('world_size', 'manifest_sha256', 'base_config_sha256', 'base_receipt_sha256'):
            if old.get(key) != contract.get(key):
                raise ValueError(f'Fork essential contract mismatch: {key}')
        keys = ('checkpoint', 'future_encoder', 'backbone_training', 'lora_rank', 'lora_alpha',
                'delta_tokens', 'action_horizon', 'action_dim', 'resampler_dim', 'resampler_heads',
                'delta_scale', 'seed', 'batch_size', 'gradient_accumulation', 'image_aug')
        for key in keys:
            if old['config'].get(key) != contract['config'].get(key):
                raise ValueError(f'Fork essential configuration mismatch: {key}')
    local = torch.load(path / f'rank_{rank}.pt', map_location='cpu', weights_only=False)
    if local['micro_step'] != metadata['progress']['micro_step']:
        raise ValueError('Rank cursor differs from checkpoint progress')
    state = torch.load(path / 'training.pt', map_location='cpu', weights_only=False)
    load_trainable_state(policy, state['model'])
    if reset_optimizer:
        optimizer.state.clear()
    else:
        hyperparameters = [{k: v for k, v in group.items() if k != 'params'}
                           for group in optimizer.param_groups]
        saved_groups = state['optimizer']['param_groups']
        if len(saved_groups) != len(hyperparameters) or any(
                old.get('group_name') != new.get('group_name')
                for old, new in zip(saved_groups, hyperparameters)):
            raise ValueError('Fork optimizer group layout differs')
        optimizer.load_state_dict(state['optimizer'])
        for group, options in zip(optimizer.param_groups, hyperparameters):
            group.update(options)
    restore_rng(local['rng'])
    return dict(metadata['progress'])

"""Native Dream E-LAF v0 joint flow training on immutable grid11k feature caches.

Checklist: complete cache identity; whole-policy DDP; mean losses; accumulation;
same-config full resume; atomic latest/best; fixed-set resampler snapshots.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from elaf.v0_model import ELAFV0Config, ELAFV0Policy
from elaf.v0_training import (atomic_json, barrier, cosine_multiplier, load_checkpoint,
    parameter_groups, sample_indices, save_checkpoint, seed_all, sha256, rng_state, restore_rng, build_scheduler,
    load_fork_checkpoint, clip_grad_groups)


def build_backbone(config):
    sys.path.insert(0, str(Path(config['dream_repo']) / 'vla'))
    from dreamvla.configuration_dreamvl import DreamVLAConfig
    from dreamvla.modeling_dreamvl import DreamVLAForActionPrediction
    from peft import LoraConfig, get_peft_model
    if not (Path(config['checkpoint']) / 'download_receipt.json').is_file():
        raise FileNotFoundError('Verified native Dream download receipt is required')
    backbone = DreamVLAForActionPrediction.from_pretrained(config['checkpoint'],
        torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', local_files_only=True)
    backbone.requires_grad_(False)
    backbone.model = get_peft_model(backbone.model, LoraConfig(r=config['lora_rank'],
        lora_alpha=config.get('lora_alpha', 16), lora_dropout=0., bias='none', target_modules='all-linear'))
    # FP32 optimizer/master parameters; autocast covers compute, base remains BF16.
    for parameter in backbone.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if config.get('gradient_checkpointing', True):
        backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    backbone.config.use_cache = False
    return backbone


def model_config(config):
    return ELAFV0Config(latent_dim=config['future_encoder']['feature_dim'],
        latent_tokens=config['delta_tokens'], resampler_dim=config['resampler_dim'],
        resampler_heads=config['resampler_heads'], action_horizon=config['action_horizon'],
        action_dim=config['action_dim'], proprio_dim=8, delta_scale=config['delta_scale'],
        action_loss_weight=config['action_loss_weight'], delta_loss_weight=config['delta_loss_weight'],
        resampler_fp32=config.get('resampler_fp32', False))


def start_monitor(output, config):
    import fcntl
    with (output / 'monitor.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None  # A previous worker still owns this run's exclusive monitor.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (output / 'monitor.log').open('a') as stream:
        process = subprocess.Popen([sys.executable, '-m', 'elaf.v0_monitor', '--run-dir', str(output),
            '--project', config['wandb_project'], '--entity', config['wandb_entity'], '--threads', '4'],
            env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    atomic_json(output / 'monitor_process.json', {'pid': process.pid})
    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--fork-from', type=Path, help='New run: model and data cursor restored, optimizer reset')
    parser.add_argument('--infra-migration', type=Path, help='Explicit old/new contract receipt for an infrastructure-only resume')
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    args = parser.parse_args()
    if args.fork_from and args.resume:
        parser.error('--fork-from and --resume are mutually exclusive')
    config = json.loads(args.config.read_text())
    rank, local_rank, world = (int(os.environ.get(key, default)) for key, default in
                              [('RANK', '0'), ('LOCAL_RANK', '0'), ('WORLD_SIZE', '1')])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 is required; no CPU fallback')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    logging.basicConfig(level=logging.INFO, format=f'%(asctime)s rank={rank} %(message)s')
    for key in ('batch_size', 'gradient_accumulation', 'max_steps', 'save_every'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be positive integer')
    if config.get('workers', 0) != 0:
        raise ValueError('Exact cursor replay currently requires workers=0')
    if rank == 0:
        if not args.resume and args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f'Output must be empty for a fresh run: {args.output}')
        args.output.mkdir(parents=True, exist_ok=True)
    barrier()
    from elaf.v0_dataset import CachedV0Dataset, collate_v0
    dataset = CachedV0Dataset(config['manifest'], config['checkpoint'], image_aug=config.get('image_aug', False))
    manifest = json.loads(Path(config['manifest']).read_text())
    if manifest.get('complete') is not True:
        raise ValueError('Training requires a complete feature cache')
    if manifest['encoder_sha256'] != config['future_encoder']['sha256']:
        raise ValueError('Cache encoder differs from selected grid11k')
    seed = config.get('seed', 239)
    seed_all(seed)
    policy = ELAFV0Policy(build_backbone(config), model_config(config)).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(policy, config['optimizer']))
    scheduler = build_scheduler(optimizer, config)
    contract = dict(config=config, world_size=world, manifest_sha256=sha256(config['manifest']),
        source_sha256={str(path): sha256(path) for path in [Path(__file__),
            Path(__file__).with_name('v0_training.py'), Path(__file__).with_name('v0_model.py'),
            Path(__file__).with_name('model.py'), Path(__file__).with_name('v0_dataset.py'),
            Path(__file__).with_name('v0_data.py'), Path(__file__).with_name('v0_cache.py'),
            Path(__file__).with_name('v0_prefetch.py')]},
        base_config_sha256=sha256(Path(config['checkpoint']) / 'config.json'),
        base_receipt_sha256=sha256(Path(config['checkpoint']) / 'download_receipt.json'))
    wrapped = DistributedDataParallel(policy, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True,
        find_unused_parameters=False) if world > 1 else policy
    seed_all(seed + rank)
    progress = dict(step=0, micro_step=0, best_loss=None, window_loss=0., window_count=0,
                    is_best=False, wandb_run_id=None)
    if args.resume:
        resume_contract = contract
        if args.infra_migration:
            migration = json.loads(args.infra_migration.read_text())
            if migration['new_contract'] != contract or Path(migration['checkpoint']).resolve() != args.resume.resolve():
                raise ValueError('Infrastructure migration does not match this source/config/checkpoint')
            resume_contract = migration['old_contract']
        progress = load_checkpoint(args.resume, policy, optimizer, scheduler, resume_contract, rank)
    if args.fork_from:
        parent = load_fork_checkpoint(args.fork_from, policy, optimizer, rank,
                                      reset_optimizer=True, contract=contract)
        progress['micro_step'] = parent['micro_step']
        if rank == 0:
            atomic_json(args.output / 'fork_receipt.json', dict(
                source=str(args.fork_from.resolve()), parent_progress=parent,
                optimizer='reset', scheduler='new recovery warmup', recovery_step=0,
                data_cursor='continued from parent', rng='restored per rank'))
    training_rng = rng_state()
    run = None
    if rank == 0:
        import wandb
        run_id = progress['wandb_run_id'] or wandb.util.generate_id()
        run = wandb.init(project=config['wandb_project'], group='elaf-v0-' + run_id,
            entity=config.get('wandb_entity'), name=args.output.name, config=contract,
            id=run_id, resume='must' if progress['wandb_run_id'] else None,
            mode=args.wandb_mode, dir=str(args.output))
        progress['wandb_run_id'] = run.id
    ids = [progress['wandb_run_id']]
    if world > 1:
        dist.broadcast_object_list(ids, src=0)
    progress['wandb_run_id'] = ids[0]
    from elaf.monitor import save_resampler_snapshot
    monitor_path = Path(config['manifest']).parent / manifest['monitor_path']
    monitor_identity = manifest['monitor_manifest_sha256']
    if rank == 0:
        atomic_json(args.output / 'resolved_config.json', contract)
        atomic_json(args.output / 'run_identity.json', dict(wandb_run_id=ids[0], parent_run_id=ids[0],
            monitor_path=str(monitor_path), monitor_manifest_sha256=monitor_identity,
            snapshot_dir=str(args.output / 'resampler_snapshots')))
        atomic_json(args.output / 'status.json', {'status': 'running', 'step': progress['step']})
        if not args.resume:
            save_resampler_snapshot(policy.resampler, args.output / 'resampler_snapshots', training_step=0,
                config=asdict(policy.config), parent_run_id=ids[0], monitor_manifest_sha256=monitor_identity)
        logging.info('E-LAF v0 parameters=%d trainable=%d hidden=%d global_batch=%d steps=%d',
            sum(p.numel() for p in policy.parameters()), sum(p.numel() for p in policy.parameters() if p.requires_grad),
            policy.hidden_size, world * config['batch_size'] * config['gradient_accumulation'], config['max_steps'])
    monitor_process = start_monitor(args.output, config) if rank == 0 and args.wandb_mode == 'online' else None
    trainable = [p for p in policy.parameters() if p.requires_grad]
    from elaf.recovery_diagnostics import parameter_partition, gradient_norms, parameter_snapshot, update_metrics, flow_metrics
    diagnostic_groups = parameter_partition(policy)
    # Debug-only independent losses prove both routes reach the shared resampler.
    # Restore all RNG afterwards so this does not change the training sample/noise.
    if not args.resume:
        # Pick actual nonterminal examples, so the Delta assertion is meaningful.
        indices = []
        for members in dataset.suite_indices.values():
            for index in members:
                if bool(dataset[index]['future_valid']):
                    indices.append(index)
                    break
        if not indices:
            raise ValueError('No valid future pair exists for gradient assertion')
        indices = [indices[rank % len(indices)]] * config['batch_size']
        debug_batch = collate_v0([dataset[index] for index in indices], pad_token_id=dataset.pad_token_id)
        debug_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in debug_batch.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            debug = policy(debug_batch)
        debug_norms = {}
        for name in ('action_loss', 'delta_loss'):
            grads = torch.autograd.grad(debug[name], list(policy.resampler.parameters()), retain_graph=True, allow_unused=True)
            value = sum(float(g.float().square().sum()) for g in grads if g is not None) ** .5
            if not value > 0 or not torch.isfinite(torch.tensor(value)):
                raise RuntimeError(f'Resampler gradient route failed: {name}={value}')
            debug_norms[name] = value
        if rank == 0:
            atomic_json(args.output / 'gradient_assertions.json', debug_norms)
            run.log({'train/grad_debug/resampler_' + name: value for name, value in debug_norms.items()}, step=0)
        del debug, debug_batch, grads
    restore_rng(training_rng)
    from elaf.v0_prefetch import BatchPrefetcher
    prefetcher = BatchPrefetcher(dataset, seed=seed, rank=rank, world_size=world,
        batch_size=config['batch_size'], start_micro_step=progress['micro_step'],
        rng_state=torch.get_rng_state(), depth=2, pin_memory=True)
    started = time.monotonic()
    initial_step = progress['step']
    while progress['step'] < config['max_steps']:
        if monitor_process is not None and monitor_process.poll() is not None:
            raise RuntimeError(f'Diagnostics worker exited ({monitor_process.returncode}); see monitor.log')
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        statistics = torch.zeros(5, device=device, dtype=torch.float64)
        accumulation = config['gradient_accumulation']
        batches = []
        counts = torch.zeros(2, device=device, dtype=torch.float64)
        data_started = time.monotonic()
        count_values = [0, 0]
        for micro in range(accumulation):
            batch, consumed_rng = prefetcher.next(progress['micro_step'] + micro)
            torch.set_rng_state(consumed_rng)
            batches.append(batch)
            count_values[0] += int(batch['action_valid'].sum()) * config['action_dim']
            count_values[1] += int(batch['future_valid'].sum()) * config['delta_tokens'] * config['future_encoder']['feature_dim']
        data_seconds = time.monotonic() - data_started
        counts = torch.tensor(count_values, device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(counts)
        for micro, batch in enumerate(batches):
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            sync = wrapped.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                result = wrapped(batch, return_debug=True)
                local_counts = torch.stack([batch['action_valid'].sum() * config['action_dim'],
                    batch['future_valid'].sum() * config['delta_tokens'] * config['future_encoder']['feature_dim']])
                weighted = torch.stack([result['action_loss'], result['delta_loss']]) * local_counts / counts.clamp_min(1)
                loss = config['action_loss_weight'] * weighted[0] + config['delta_loss_weight'] * weighted[1]
                (loss * world).backward()  # DDP averages across ranks; element means span the full update.
            statistics[:3] += torch.stack([loss.detach(), weighted[0].detach(), weighted[1].detach()]).double()
            statistics[3] += (result['raw_delta'].detach().float().square() * batch['future_valid'][:, None, None]).sum().double() / counts[1].clamp_min(1)
            statistics[4] += (batch['actions'].detach().float().square() * batch['action_valid'][:, :, None]).sum().double() / counts[0].clamp_min(1)
            progress['micro_step'] += 1
        if world > 1:
            dist.all_reduce(statistics)
        debug_interval = config.get('recovery_diagnostics_every', 0)
        debug_update = rank == 0 and debug_interval > 0 and (progress['step'] % debug_interval == 0)
        extra_metrics = {}
        if debug_update:
            extra_metrics.update(gradient_norms(diagnostic_groups))
            extra_metrics.update(flow_metrics(result, batch))
            before_update = parameter_snapshot(diagnostic_groups)
        if config['optimizer'].get('clip_mode', 'global') == 'group':
            clip_report = clip_grad_groups(optimizer, config['optimizer']['max_grad_norm'])
            norm = clip_report['global_preclip']
        else:
            norm = torch.nn.utils.clip_grad_norm_(trainable, config['optimizer']['max_grad_norm'], error_if_nonfinite=True)
        gradient_tensors = {}
        for group in optimizer.param_groups:
            name = group['group_name']
            grads = [p.grad for p in group['params'] if p.grad is not None]
            if grads:
                squared = torch.stack(torch._foreach_norm(grads, 2)).square().sum()
                gradient_tensors[name] = gradient_tensors.get(name, 0.) + squared
        gradient_values = torch.stack(list(gradient_tensors.values())).sqrt().tolist()
        gradients = dict(zip(gradient_tensors, gradient_values))
        optimizer.step()
        if debug_update:
            extra_metrics.update(update_metrics(diagnostic_groups, before_update))
            del before_update
        scheduler.step()
        progress['step'] += 1
        progress['window_loss'] += statistics[0].item()
        progress['window_count'] += 1
        step = progress['step']
        record = dict(step=step, **{f'train/{k}': statistics[i].item() for i, k in enumerate(('loss', 'action_loss', 'delta_loss'))},
            **{f'train/grad/{key}': value for key, value in gradients.items()},
            **{'train/grad/global_preclip': float(norm), 'train/scale/delta_raw_rms': statistics[3].sqrt().item(),
               'train/scale/delta_target_rms': statistics[3].sqrt().item() / config['delta_scale'],
               'train/scale/action_rms': statistics[4].sqrt().item(),
               'train/data_wait_seconds': data_seconds,
               'train/seconds_per_update': (time.monotonic() - started) / (step - initial_step)})
        for group in optimizer.param_groups:
            record['train/lr/' + group['group_name']] = group['lr']
        record.update(extra_metrics)
        if rank == 0:
            run.log(record, step=step)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            if step <= 3 or step % 10 == 0:
                logging.info('step=%d/%d loss=%.6f action=%.6f delta=%.6f seconds/update=%.3f',
                    step, config['max_steps'], *statistics[:3].tolist(), record['train/seconds_per_update'])
            if step in config['diagnostics']['steps'] or step % config['diagnostics']['interval_after_1000'] == 0:
                save_resampler_snapshot(policy.resampler, args.output / 'resampler_snapshots', training_step=step,
                    config=asdict(policy.config), parent_run_id=ids[0], monitor_manifest_sha256=monitor_identity)
        pause = torch.tensor(int((args.output / 'PAUSE_REQUEST').exists()), device=device)
        if world > 1:
            dist.all_reduce(pause, op=dist.ReduceOp.MAX)
        stopped = bool(pause.item()) or (args.stop_after_updates is not None and step >= args.stop_after_updates)
        scheduled = step % config['save_every'] == 0 or step == config['max_steps']
        if scheduled or stopped:
            progress['is_best'] = False
            if scheduled:
                window = progress['window_loss'] / progress['window_count']
                progress['is_best'] = progress['best_loss'] is None or window < progress['best_loss']
                if progress['is_best']:
                    progress['best_loss'] = window
                progress['window_loss'], progress['window_count'] = 0., 0
            save_checkpoint(args.output / 'checkpoints', policy, optimizer, scheduler, progress, contract, rank)
        if stopped:
            break
    prefetcher.close()
    if rank == 0:
        atomic_json(args.output / 'status.json', {'status': 'completed' if progress['step'] == config['max_steps'] else 'paused',
                    'step': progress['step']})
        if (args.output / 'PAUSE_REQUEST').exists():
            (args.output / 'PAUSE_REQUEST').unlink()
        run.finish()
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if int(os.environ.get('RANK', '0')) == 0 and '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            if (output / 'run_identity.json').exists():
                atomic_json(output / 'status.json', {'status': 'failed'})
        raise

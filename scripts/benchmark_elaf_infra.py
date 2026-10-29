"""Model-only native E-LAF throughput; isolated from production and W&B."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from elaf.v0_train import build_backbone, model_config
from elaf.v0_model import ELAFV0Policy
from elaf.v0_dataset import CachedV0Dataset, collate_v0
from elaf.v0_training import parameter_groups, seed_all, sample_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpointing', choices=['on', 'off'], required=True)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--fused-optimizer', action='store_true')
    parser.add_argument('--save-trainables', type=Path)
    parser.add_argument('--sample-step', type=int, help='Use production mixed sampling at this micro step')
    parser.add_argument('--bucket-cap-mb', type=float, default=25)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config['gradient_checkpointing'] = args.checkpointing == 'on'
    seed_all(config.get('seed', 239))
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        torch.distributed.init_process_group('nccl')
    report = dict(checkpointing=args.checkpointing, batch_size=config['batch_size'],
                  warmup=args.warmup, iterations=args.iterations, status='running', fused_optimizer=args.fused_optimizer,
                  rank=rank, world_size=world, sample_step=args.sample_step, bucket_cap_mb=args.bucket_cap_mb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset = CachedV0Dataset(config['manifest'], config['checkpoint'], image_aug=config.get('image_aug', False))
        indices = list(range(config['batch_size'])) if args.sample_step is None else sample_indices(
            dataset.suite_indices, seed=config.get('seed', 239), micro_step=args.sample_step,
            rank=rank, world_size=world, batch_size=config['batch_size'])
        samples = [dataset[i] for i in indices]
        batch = collate_v0(samples, pad_token_id=dataset.pad_token_id)
        report.update(sample_indices=indices, input_ids_shape=list(batch['input_ids'].shape))
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
        policy = ELAFV0Policy(build_backbone(config), model_config(config)).cuda().train()
        wrapped = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank],
            broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=False,
            bucket_cap_mb=args.bucket_cap_mb) if world > 1 else policy
        seed_all(config.get('seed', 239) + rank)
        optimizer = torch.optim.AdamW(parameter_groups(policy, config['optimizer']), fused=True if args.fused_optimizer else None)
        durations = []
        segments = {key: [] for key in ('forward', 'backward', 'clip', 'optimizer')}
        losses = []
        torch.cuda.reset_peak_memory_stats()
        for index in range(args.warmup + args.iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            events[0].record()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = wrapped(batch)
            events[1].record()
            result['loss'].backward()
            events[2].record()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config['optimizer']['max_grad_norm'], error_if_nonfinite=True)
            events[3].record()
            optimizer.step()
            events[4].record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if index >= args.warmup:
                durations.append(elapsed)
                for j, key in enumerate(segments):
                    segments[key].append(events[j].elapsed_time(events[j + 1]) / 1000)
            losses.append(float(result['loss']))
            print(json.dumps(dict(iteration=index, seconds=elapsed, loss=float(result['loss']))), flush=True)
        report.update(status='passed', seconds=durations, mean_seconds=sum(durations) / len(durations),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      segment_seconds=segments, mean_segment_seconds={k: sum(v)/len(v) for k,v in segments.items()}, losses=losses)
        if args.save_trainables:
            from elaf.v0_training import trainable_state
            if rank == 0:
                torch.save(trainable_state(policy), args.save_trainables)
    except torch.cuda.OutOfMemoryError as error:
        report.update(status='oom', error=str(error), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    output = args.output if world == 1 else args.output.with_name(args.output.stem + f'.rank{rank}' + args.output.suffix)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()

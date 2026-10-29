"""Explicit low-LR schedule for extending an already completed action run."""
import math


def continuation_lr(step, end, config):
    start = int(config['start_step'])
    warmup = int(config['warmup_steps'])
    if not start < end or not 0 < warmup < end - start:
        raise ValueError('Invalid continuation interval')
    elapsed = max(0, min(step - start, end - start))
    initial, peak, final = (float(config[k]) for k in ('initial_lr', 'peak_lr', 'final_lr'))
    if elapsed < warmup:
        return initial + (peak - initial) * elapsed / warmup
    fraction = (elapsed - warmup) / (end - start - warmup)
    return final + .5 * (peak - final) * (1 + math.cos(math.pi * fraction))

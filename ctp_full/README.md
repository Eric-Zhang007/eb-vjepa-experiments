# Full DROID Continued Pretraining

Independent experiment: `experiments/ctp_full_72h`. Never resumes or overwrites
CTP-v1 or the LIBERO action-only experiment. It starts from the official
distilled ViT-L checkpoint, aligning both encoders with `ema_encoder`.
The copied CTP-v1 implementation is isolated to preserve earlier run contracts.

## Data Contract

The download inventory contains 813 videos. The complete 95658-episode metadata
references 784 of them, with no missing referenced file. The 29 unreferenced
files remain untouched and are explicitly listed in `data_report.json`; their
episode boundaries and outcome labels must not be guessed.

The fixed 16-frame / 4 Hz clip requirement excludes 1739 short episodes:
130 successful and 1609 unsuccessful. There are 93919 eligible episodes,
each with three camera views, yielding 281757 episode-camera samples.
Sampling selects a random legal clip within an episode, without crossing its
boundary or substituting a different sample on decode error.

`is_episode_successful` is explicit dataset metadata. Its per-episode minimum
and maximum must agree and be boolean. It is used only by the sampler:
no action, language, proprioception, reward, task labels, or outcome labels
enter model inputs or JEPA supervision.

Original episode counts: success 78864, failure 16794. Global batches preserve
this original ratio (82.4437057% success), not the changed ratio after short
episode filtering. Each 128-clip global batch has 105 or 106 successful clips
and 23 or 22 failed clips. Integer cumulative rounding keeps aggregate counts
within half a sample of the requested ratio. Within each outcome class,
episode-camera indices are shuffled, consumed, then reshuffled.
Ranks partition a common shuffled global batch; accumulation is supported.
Resume reconstructs the same indices and per-sample augmentation seeds.

## Training Contract

- 8-GPU NCCL DDP, BF16/SDPA, online encoder and predictor trainable.
- Frozen EMA target, momentum 0.99925, seed 239, effective batch 128.
- Official video transforms, masks, and progressive prediction/context loss.
- AdamW, peak LR 2e-5, 500-step warmup from 2e-6, WD 0.04.
- After warmup, cosine follows elapsed wall time to 2e-6 at 72 hours.
  Rank zero broadcasts the schedule clock, preventing per-rank LR differences.
- One-million-step safety ceiling replaces the old 30k limit.
- Same explicit checkpoint adaptation as pilot: 12-block predictor trunk fully
  reused; only two 1664-dim teacher-space output heads become 1024-dim heads.
  Reports include all missing/unexpected keys, shape checks and reuse fractions.
- Latest resumable checkpoint every 500 steps; permanent checkpoints/EMA exports
  every 5000; additional wall-clock 24h/48h/72h snapshots and final.
- Wall time includes initialization/checkpoint overhead and downtime on resume.
- Less than 12 GiB free at a periodic checkpoint boundary triggers a final save
  and safety stop, never silent deletion of previous experiment artifacts.

## Acceptance and Launch

```bash
ROOT=/data/users/jiachangzhang/eb_vjepa
export TMPDIR=$ROOT/cache/tmp
$ROOT/envs/eb_vjepa_pretrain/bin/python $ROOT/ctp_full/prepare.py
$ROOT/envs/eb_vjepa_pretrain/bin/python $ROOT/ctp_full/test_sampling.py
$ROOT/envs/eb_vjepa_pretrain/bin/python $ROOT/ctp_full/start.py --detach
```

Preparation refuses to overwrite an existing frozen manifest.
The launch supervisor holds an exclusive lock and records its PID and phase
in `pipeline_status.json`. An existing status or production run must be
inspected explicitly, never overwritten to bypass the duplicate guard.

The bounded calibration reuses pilot-established microbatch 16/GPU and only
compares activation checkpointing ON/OFF, 40 steps each. A single 100-step
smoke run is split 90+10 for real save/load/resume, then formal training starts
automatically only if all gates pass and the GPUs are free.

Online W&B project is `eb-vjepa`, formal name
`droid_vitl16_256_16f_ctp_full_72h`. Formal command/PID, selected throughput,
and run identity are written to the experiment's `RUN.md`, `launch.json`,
and `wandb.json`. `metrics.jsonl` includes actual per-global-batch outcome
counts, encoder gradients, EMA updates, context-loss weight, elapsed time,
loss components and performance.

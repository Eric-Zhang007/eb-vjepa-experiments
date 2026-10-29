# CTP-v1

Single-node, eight-GPU, video-only DROID continued pretraining using upstream
V-JEPA 2.1 modules. This directory does not modify the upstream repository.

## Initialization

Both online and target encoders initialize strictly from `ema_encoder`, matching
the official distilled ViT-L evaluation convention. The released predictor is
first instantiated as 12 blocks, eight mask tokens, 384 hidden dimensions,
1024 input dimensions and 1664 output dimensions, and loaded with `strict=True`.
Missing/unexpected keys and shape mismatches stop the run.

The release predicts ViT-G teacher features. For native ViT-L EMA JEPA, the
explicit `predictor_head_policy: reinitialize_native` replaces ONLY
`predictor_proj` and `predictor_proj_context` with 1024-output linear layers.
The pretrained input projection, all 12 transformer blocks, norms, embeddings,
and mask tokens are retained. Retained released predictor parameter fraction:
94.4227%; transformer trunk: 100%. The report distinguishes the initial full
released-model load from this declared head adaptation. It never silently
uses `strict=False` or discards a random subset of tensors.

The target is the final 1024-dimensional encoder feature, not four concatenated
hierarchical levels. The L1 prediction and distance-weighted context objectives
come from upstream. Its context coefficient ramps from 0 to 0.5 between steps
15,000 and 30,000. No teacher model, image branch, robot action, language,
proprioception, reward or task labels enter the objective.

## Fixed Recipe

- 256 pixels, 16 frames, timestamp-based 4Hz sampling, patch 16, tubelet 2.
- Official video augmentation and both official multi-block masking groups.
- BF16, SDPA, AdamW, no LR scaling, weight decay 0.04 (bias/norm excluded).
- LR: 500-step linear warmup from 2e-6 to 2e-5, cosine to 2e-6 at step 30,000.
- EMA 0.99925; seed 239; global batch exactly 128.
- Stop at 30,000 optimizer steps or 72 wall-clock hours from formal startup.
  The wall-clock deadline survives resume, including downtime.
- Calibration selects microbatch/accumulation/checkpointing, not learning rate.

The frozen episode manifest contains the available completed videos, not the
entire DROID download. Current snapshot: 154 exterior_1_left files, 48,055
eligible episode-camera samples. Future snapshots can include all three cameras.
Do not regenerate the manifest during a run: its SHA256 is a resume contract.

## Start

On the server:

```bash
ROOT=/data/users/jiachangzhang/eb_vjepa
export TMPDIR="$ROOT/cache/tmp"
export PYTHONPATH="$ROOT/repos/vjepa2"
"$ROOT/envs/eb_vjepa_pretrain/bin/python" "$ROOT/ctp_v1/launch.py" --launch
```

The launcher refuses an incomplete released checkpoint or occupied GPUs.
It runs sequential trials only: microbatches 4/8/16, activation checkpointing
ON/OFF, 40 optimizer steps per candidate (5 startup + 35 measured). Larger
batches in an OOM/headroom-limited mode are skipped. Total calibration budget
is one hour, with at most six trials and no retries/sweep of the optimizer
recipe. It selects the fastest stable candidate below 90% allocated GPU memory.
All candidates use effective batch 128 and DDP `no_sync()` during accumulation.

Then it runs 90 smoke steps, saves, restores, and runs steps 91-100.
This is one 100-step smoke with a real optimizer/RNG/EMA resume check.
Each step checks finite loss/gradients, nonzero encoder gradient, no target
gradient, and EMA changes. Evidence includes eight-rank checks, loss variation,
actual DROID video provenance, and W&B logs.

Only after these gates pass does `--launch` start the background formal
`torchrun --standalone --nproc_per_node=8` process, observe its first metric,
and record command, PID/session leader, W&B ID, time, throughput and stop
conditions in `experiments/ctp_v1/RUN.md`. Without `--launch`, it stops after
the gates. No unattended launcher has been installed as a service or timer.

## Checkpoints And Resume

- `latest.pt`: atomic resumable save every 250 steps and at termination.
- `step_005000.pt`, etc.: immutable hard-link snapshots every 5k steps.
- `final.pt`: last resumable state, including wall-clock-cap termination.
- `embodied_vjepa_encoder_XXXXXX.pt`: EMA target only, with selection metadata.

The resumable state includes online encoder, predictor, `ema_encoder`, AdamW,
LR scheduler step, optimizer step, clips seen, per-rank Python/NumPy/CPU/CUDA
RNG, source-load report, W&B ID, wall-clock start and configuration contract.
The data sampler recreates rank partitions, episode windows and augmentations
from the saved optimizer step; prefetched samples do not advance saved state.
Mask generation is deterministic in the main process, not worker-local queues.
Changes to core implementation, batch, schedule or manifest fail resume.

Resume a formal run explicitly (do not rerun calibration):

```bash
"$ROOT/envs/eb_vjepa_pretrain/bin/torchrun" --standalone --nproc_per_node=8 \
  "$ROOT/ctp_v1/train.py" --config "$ROOT/experiments/ctp_v1/calibrated.yaml" \
  --stage train --resume "$ROOT/experiments/ctp_v1/latest.pt"
```

W&B uses the existing installation. Credentials present: online; otherwise
offline. Logs/caches stay on the data disk; checkpoints are never uploaded
as W&B artifacts. Per-step JSONL also records losses, LR, grad norms, EMA,
clips, timings and memory. Backward timing includes overlapped DDP all-reduce;
it is not falsely labeled a separately measured communication time.

## Verification Status

Passed:

- Five CPU tests covering LR boundaries, strict reports, EMA, distributed
  sample partitions, atomic save and optimizer resume.
- Exact real DROID RGB/augmentation replay and episode-boundary rejection.
- Released metadata versus model-definition alignment: no missing/unexpected
  keys or shape differences against `ema_encoder` and the released predictor.
- One random-weight ViT-L GPU architecture unit test: BF16 forward/backward,
  official masks/loss, nonzero encoder gradients, no target gradients.
- Eight-rank tiny-model regression: `no_sync`, identical reduced gradients,
  bit-exact resumed next step (model/optimizer/EMA/RNG), W&B offline writing.

Not yet verified: full released tensor loading, real pretrained 100-step smoke,
bounded throughput calibration, online W&B connectivity, or a formal run.
The released checkpoint was still downloading during implementation.
Random-weight unit tests are not claimed as pretrained training evidence.

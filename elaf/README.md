# E-LAF development snapshot

2026-10-15. First implementation, not a trained policy or a benchmark result.

## What exists

- `model.py`: native Dream-VLA image encoder and shared bidirectional decoder;
  scalar action tokens and compact future-feature tokens with independent flow times.
  Loss is `action_mean_mse + lambda_latent * latent_mean_mse`.
- `prepare_data.py`: **small real-data smoke preparation only**. Uses all 1692 eligible
  LIBERO training episodes for action/proprio mean/std; excludes Goal episode82.
  Samples use current external camera and frames t+7,t+8 as the future target.
  Frozen grid11k EMA V-JEPA encodes at 384 pixels; 24x24 patches are averaged to 4x4
  and feature-normalized without learnable parameters. These are predictive visual
  targets; their physical meaning and this short temporal window remain hypotheses.
- `data.py`: verifies preparation provenance/checksums and collates native image patches.
- `train.py`: frozen vision/text embeddings, LoRA on native decoder attention by default,
  both new stream adapters trainable; Accelerate state includes optimizer and RNG.
  Full-backbone training is explicit and has not been sized for the server.

## Server commands

Run from `/data/users/jiachangzhang/eb_vjepa`:

```bash
envs/elaf/bin/python -m elaf.prepare_data \
  --output experiments/elaf/data_smoke_grid11k --max-samples 4
ELAF_TEST_NATIVE=1 PYTHONPATH=$PWD/repos/Dream-VLX/vla \
  envs/elaf/bin/python -m unittest elaf.test_model elaf.test_data -v
envs/elaf/bin/python -m elaf.train --config elaf/configs/smoke.json \
  --output experiments/elaf/tiny_smoke
envs/elaf/bin/python -m elaf.train --config elaf/configs/smoke.json \
  --output experiments/elaf/tiny_resume \
  --resume experiments/elaf/tiny_smoke/checkpoints/step_000001
```

The smoke config uses a **random tiny native Dream model**, real images/actions and
the actual grid11k teacher. It validates plumbing, not downloaded-model performance.
Do not interpret it as Dream-VLA fine-tuning. Production loading requires the completed
weight `download_receipt.json`. `configs/libero.json` deliberately points at a full-data
manifest that does not yet exist, so a four-sample smoke set cannot silently start a
long training run. A full parallel cache/data pipeline remains to be implemented after
choosing camera count and temporal target. The smoke cache is not scalable production I/O.

Joint inference updates both streams for exactly four Euler steps by default.
`fixed_latent_noise` keeps the original latent noise and u=0, while updating actions.
The training mixture places 10% probability at u=0. This is a fixed-noise control,
not cached future prediction. `action_only` requires the corresponding trained
action-only architecture; it is an E-LAF ablation, not an exact reproduction of the
official Dream-VLA finetuning baseline.

Inputs are clean at every flow step, but their internal hidden states are recomputed
because attention is bidirectional. Decoder KV caching is therefore disabled.

## Environment and provenance

Isolated `envs/elaf` overlays transformers4.51.3, peft0.17, accelerate1.10.1,
tokenizers0.21.4, numpy1.26.4 and pyarrow17 on the existing pretraining environment's
PyTorch2.11/CUDA12.8 packages via a `.pth` path. It does not alter that environment.
This shared dependency path must exist; this is not a standalone relocatable environment.

Official source: `DreamLM/Dream-VLX`, commit `a72fa4106eae97a923b531342fb692742809782f`.
Model: `Dream-org/Dream-VLA-7B`, revision `0cd0e25c1771c5b468c74585b5cf35c4a4aa315c`.
Teacher: grid encoder11000, SHA256
`468d7af3a5bdae9fb2205718ab0f86a13f5395afcc53700500439d993baf987a`.

See `experiments/ELAF_ARCHITECTURE_AUDIT.md`, `ELAF_DOWNLOAD.md`, and
`ELAF_ENVIRONMENT.md` for source inspection and background preparation status.
No full fine-tuning or benchmark rollout was started during this setup.

## Verified this session

All 11 model/data tests passed, including native Dream visual+decoder CUDA BF16
forward/backward. Four real samples were generated in `data_smoke_grid11k`.
Two-step training completed under `experiments/elaf/tiny_smoke_v2`; resuming its
step1 into `tiny_resume_v2` reproduced step2 exactly. Both final safetensors SHA256:
`c10ca48e2cedf1dcd2e10d348b43d6952149ff585c8af4236f1a2f1ddca3257b`.
Machine-readable evidence: `experiments/elaf/resume_verification.json`.
Multi-GPU training and full 8.3B checkpoint loading have not yet been tested.

# Grid11k + predictive-context latent WM

Authorized on 2026-10-21. Independent experiment; E-LAF was safely paused at
42,015 updates with full state retained. This experiment does not use Dream-VLA,
LoRA, an E-LAF resampler, or moving latent targets.

## Model and information flow

Adapt V-JEPA Policy's current-position K/V interface to the existing
RoboActualizer DiT-S, rather than reproducing its full model or training recipe.
Reference code is in `references/vjepa_policy_20261021/`, with inspected revision
recorded in `source.json`. Its public predictor implementation uses unrestricted
attention between current features and future learned mask tokens.

1. Freeze the grid11k V-JEPA encoder. Retain the existing two-camera 224x448,
   four-frame, interval-two preprocessing and last-tubelet 392 spatial tokens.
2. Normalize with the exact original grid11k channel statistics, and retain the
   learned 1024-to-384 projection from the action-only model.
3. Append four complete spatial lattices of learned mask queries: 4x392 tokens.
   Query content is shared; existing 3D RoPE encodes spatial positions and
   future offsets +4,+8,+12,+16. No target features enter these queries.
4. Current tokens and future queries interact in the 12 image-expert blocks.
   Language and proprioception remain cross-attention conditions.
5. Each layer exports only the current 392 positions' post-RoPE K/V. The
   corresponding action-expert layer reads these K/V and its own action tokens.
   Actions do not read future-position K/V; the predictor does not read actions.
6. Regress frozen encoder features at the four future endpoints. Retain the
   original shifted-sigma action flow objective, with a 16x7 action chunk.

The loss is `action_flow_mse + future_latent_mse`, both using the existing masked
per-sample reduction. Future prediction is a single deterministic forward pass,
not a second flow. Its latent loss is therefore not comparable numerically to
E-LAF Delta Flow loss. Zero prediction and copying the current latent are logged
as reference MSEs using exactly the same padding reduction.

At inference the predictor runs once per observation, and its current-position
K/V is reused across action denoising steps. Future decoder output is unnecessary
for action generation, but the learned future queries remain part of the predictor.

## Initialization and training

- Source: grid encoder 11,000; action-only head 20,000.
- Reuse all 468 existing non-backbone checkpoint tensors, including normalization.
- New tensors: one shared future query and the latent decoder's modulation,
  weight and bias. Future output linear layer starts at zero.
- Fresh AdamW state and schedule; new experiment step starts at zero. This is
  additional training from an existing action head, not a from-scratch matched run.
- GPU 0-5, micro-batch 40, accumulation 1, global batch 240. Original action-only
  training used global batch 256; this difference is explicit.
- 20,000 additional optimizer updates; AdamW LR 2e-4, betas (.9,.95), WD .01.
- Existing 5% warmup, cosine decay to 2e-6, global clipping 1.0.
- FP32 master weights; BF16 autocast. Predictor checkpointing disabled after
  measuring the actual model's memory/throughput.
- Save weights every 5,000 updates; existing full-state latest/previous rotation.
- W&B training project `eb_vjepa`; evaluation never uploads to W&B.

The same 1,692 selected demonstrations and original action/state normalization
are retained. The retired grid11k per-frame cache is rebuilt in this experiment's
own directory. The old checkpoint, logs, metadata and statistics are preserved.

## Verification

`scripts/test_predictive_context.py` checks target leakage, action gradients to
queries, current-only exported K/V, checkpointed gradient equivalence, masked
losses, baseline reduction, and cached/uncached action generation.

Real-weight preflight reuses a LIBERO sample, checks target independence, confirms
nonzero Action-to-query gradient, performs two optimizer updates on a disposable
copy, and compares uncached, cached and CUDA-graph inference. Production starts
from the initialization saved before those disposable updates.

The supervisor then runs a six-GPU pause/resume test on real training data:
pause after update 1, restore full optimizer/RNG/sampler state, finish at update 3,
and check optimizer step and data cursor. It launches production only after this
test passes. Evidence is in `preflight/results.json`, `unit_tests.log`,
`ddp_resume_test.json`, and the smoke logs. Their existence/results determine
completion; this README is a design record, not a live status claim.

## Implementation and operation

New policy: `RoboActualizer/src/Roboactualizer/models/roboactualizer/predictive_context.py`.
Existing factory selects it with `model.predictive_context=true`; existing trainer
and data pipeline are reused. Trainer changes make joint training enter DDP
`forward`, extend strict resume checks to this variant, support safe pause, and
correct throughput accounting with gradient accumulation. Dataset strict mode
prevents random sample substitution after a read failure.

Configuration: `train_config.yaml`. Remote root:
`/data/users/jiachangzhang/eb_vjepa`.

```bash
PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/src:/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages \
/data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python \
/data/users/jiachangzhang/eb_vjepa/scripts/run_predictive_context_experiment.py \
  --root /data/users/jiachangzhang/eb_vjepa \
  --output /data/users/jiachangzhang/eb_vjepa/experiments/grid11k_latent_wm_20261021
```

The launched process is already recorded in `launch.json`; do not duplicate it.
Top-level `status.json` records the current phase. Production logs are `train.log`
and `train/train_metrics.jsonl`. To pause production, create `train/PAUSE_REQUEST`.

After 20K additional updates, evaluate clean200 and Plus280 with the corrected
grid11k comparison protocol: four denoising steps, execute ten actions, settle
thirty steps, native 400/400/400/700 horizons, canonical clean language, and
original Plus perturbed instructions. Evaluation uses GPUs 0 and 1 after training
exits. This is the same exploratory protocol as the grid11k comparator, not a
claim of matching OpenPI's default full evaluation.

Any success-rate improvement over the existing 20K action-only head is preliminary:
the new run has additional training updates and a slightly different global batch.
A matched continued-action-only control would be needed to isolate the WM benefit.

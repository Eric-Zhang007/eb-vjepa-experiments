# Parallel closed-loop evaluation

## Model identities

`roboactualizer_action_only` loads the existing action-only derivative of
RoboActualizer. Its V-JEPA encoder is a component, not a separate policy and not
E-LAF. Official RoboActualizer joint checkpoints are deliberately rejected by this
adapter. `elaf` loads the new Dream-VLA-based action/latent policy, with its own
training configuration, normalization and teacher provenance. Tiny test checkpoints
are recorded as `tiny=true`; they are only pipeline checks.

## Architecture

Each selected GPU has one model service, with independent simulator subprocesses.
Ready observations are batched with a bounded collection window; a slow reset does
not hold up all other environments. The service can also merge requests from separate
XPolicyLab bridges. Each simulator executes the explicitly configured action prefix,
updates history at every control step, and requests its next action chunk. No tensor
parallelism or model replication per simulator is used.

The service listens only on localhost and authenticates clients using a generated
0600 key file. The pickle transport is for trusted local processes only. Different
simulator and model Python environments are supported. Memory copies across the local
socket are included in observed rollout throughput.

## LIBERO / Plus / Pro

Run from `/data/users/jiachangzhang/eb_vjepa`. Manifests are explicit lists of unique
`benchmark,suite,task_id,trial` identities. Existing exploratory sets contain clean200,
Plus280 and a Pro manifest; the latter is not a validated policy score. Never aggregate
different manifests or policy hashes into one result.

Create a fresh selection with `python -m elaf.evaluation.manifest --benchmark clean
--output <path>` (default five trials per task), or pass `--source` for a saved
Plus/Pro selection. `--limit` explicitly creates a smoke/throughput subset.

```bash
envs/elaf/bin/python -m elaf.evaluation.launch \
  --policy-config elaf/evaluation/configs/roboactualizer_smoke.json \
  --manifest experiments/elaf/evaluation_manifests/clean.json \
  --benchmark clean --gpus 0,1,2,3 --environments 8 --batch-size 8 \
  --execute-steps 10 --output experiments/elaf/eval_example
```

This example uses the **old RoboActualizer action-only** policy, encoder11000,
action20000; it is not E-LAF evaluation. For an E-LAF trained checkpoint, use
`{"type":"elaf","checkpoint":".../checkpoints/step_XXXXXX","num_steps":4,"mode":"joint"}`
and the corresponding trained protocol (`execute-steps=8` for current LIBERO setup).
The supplied `elaf_tiny.json` is only a plumbing test.

The simulator retains original image orientation, external and wrist cameras, state
conversion, initial states, 30 settling steps, 400/700-step horizons and gripper
conversion. The old model retains both cameras and temporal history. E-LAF currently
uses its trained single external-camera input. Noise is derived from episode identity
and replan index so request ordering does not change the noise draw. BF16 batch shapes
can still cause small numerical differences and diverging closed-loop trajectories;
do not promise bitwise-identical episode outcomes across batch sizes.

Results include per-episode records, policy/checkpoint/encoder/statistics fingerprints,
execution protocol, completed counts, time, steps/s and actual inference batch sizes.
Failures and timeouts are errors rather than failed episodes. A complete summary is
published only after exact unique episode coverage passes. Each worker has a timeout;
RPC inference has its own timeout. Cleanup targets only child processes of this run.

## Throughput selection

```bash
envs/elaf/bin/python -m elaf.evaluation.sweep \
  --policy-config elaf/evaluation/configs/roboactualizer_smoke.json \
  --manifest experiments/elaf/evaluation_manifests/clean_throughput16.json \
  --benchmark clean --gpu 0 --execute-steps 10 --sizes 1,4,8 \
  --output experiments/elaf/throughput_example
```

The sweep runs configurations sequentially with the same manifest and checkpoint.
Use at least as many episodes as the largest concurrency; for a definitive choice,
repeat with enough tasks and trials to reduce startup and easy-task effects. Main
metric is completed valid episodes/hour. Both model-load-inclusive and rollout-only
rates are recorded; rollout-only still includes environment startup/reset, which is
real work. `environments` and `batch-size` are separate controls. An OOM fails the run;
there is no silent batch or protocol adjustment. No single concurrency is assumed
best for both the old policy and full Dream-VLA/E-LAF.

The local transport bounds response wait, but connection/authentication/send are
trusted-local blocking operations, not a hardened remote serving protocol. The
launcher owns the local service lifecycle. Do not expose its port outside localhost.

## RoboTwin / RoboDojo

Native XPolicyLab adapters and launch scripts are in `scripts/robodojo/elaf_policy`
and `scripts/robodojo/launch_*.sh`. Details: `experiments/ELAF_SIM_INTERFACES.md`.
They preserve environment indices and use the same shared inference service. For
Isaac Sim, use its native vectorized environments rather than blindly launching
many complete simulator applications per card.

Current LIBERO7D weights are not compatible with dualarm14/16D control. E-LAF inference
can accept a future checkpoint with matching explicitly declared action/state layout;
there is no implicit padding or coordinate conversion. These robot-specific policies
have not been trained. RoboDojo's simulator installation/assets are still a background
task, so no RoboDojo throughput or task success has been claimed.

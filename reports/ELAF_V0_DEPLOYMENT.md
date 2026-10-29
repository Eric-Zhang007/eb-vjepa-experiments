# E-LAF v0 queued deployment — 2026-10-16

Remote root: `/data/users/jiachangzhang/eb_vjepa`.

- Encoder: grid11k, hash in `ELAF_V0_ENCODER_RECEIPT.json`.
- CPU RLDS export PID2643374; feature supervisor PID2804679.
- Cache directory: `data/elaf_v0_grid11k_raw`; export.log, supervisor.json,
  worker_0.log and worker_1.log record progress. Completion is published only
  after all episode payload hashes are checked.
- Watcher PID2976024, verified alive and `waiting_cache` after deployment.
- Watcher plan/status/log: `experiments/elaf_v0_watch_20261016/`.
- Formal training output: `experiments/elaf_v0_grid11k_72token`.
- W&B project: `eb_vjepa`, separate E-LAF training and diagnostic runs in one group.

All47 CPU tests passed, including native Dream attention/gradient checks.
Actual grid11k forward produced finite BF16 [2,1152,1024] features; a real cached
sample passed tiny-native Dream LoRA forward/backward. Full-sized GPU preflight
is queued, not claimed complete.

Sequence: source export → raw features on two genuinely idle cards → complete
cache → GPUs0–5 idle continuously30s → full-model two-GPU training/resume test →
GPUs0–5 idle continuously30s → six-GPU200k training. The watcher checks no compute
process, memory below512MiB, utilization at most5%, sampled every5s. It never
terminates existing jobs. Failures block launch and are recorded; no automatic
retry can duplicate the training job. It continues watching the launched process.

Training keeps latest and lowest-window-training-loss checkpoints every5000
updates. Four standard representation diagnostics consume snapshots on CPU.
GPU6/7 were occupied by another user's jobs at deployment. Both raw-feature
generation and full-model validation therefore remain pending GPU availability.

Source/config hashes are sealed in plan.json. Editing covered files after arming
blocks launch; deliberate future changes require reviewing and replacing the
plan, never simply deleting launch claims. PIDs above are historical receipts;
query live status/process identity before operational actions.

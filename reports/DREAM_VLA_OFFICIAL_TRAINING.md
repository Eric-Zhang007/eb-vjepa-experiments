# Dream-VLA mixed LIBERO training

Current target: one model trained on all four LIBERO suites, 200000 optimizer
updates total. The official Dream-VLX training entrypoint and model/loss are
retained. This is a mixed-suite experiment, not reproduction of the published
shell's four independent runs.

Production launched 2026-10-16 00:50 server time, PID939379, W&B run
`mrmlszkq` in `eb_vjepa`. Launcher/log/source snapshot:
`experiments/dream_vla_mix4/launch_20261016_005020/`.

## Fixed experiment settings

- GPUs0-5 only; GPUs6-7 reserved for other work.
- Batch8 per GPU, accumulation1, global batch48 throughout.
- Four-suite sampling:25% each (Spatial, Object, Goal, LIBERO-10).
- Official RLDS data, two images, proprioception, eight-action flow matching.
- LoRA rank32/alpha16, learning rate1e-4, decay at100000 updates.
- W&B project `eb_vjepa`; output `experiments/dream_vla_mix4`.
- Save every5000 updates and at completion; retain latest and lowest window
  mean training-loss candidate. This is not validation-success selection.
- Full resume and official-checkpoint Plus evaluation must pass real tests
  before the new production launch. See the verification report for results.

## Assets and implementation

Source DreamLM/Dream-VLX commit a72fa4106eae97a923b531342fb692742809782f.
Base Dream-org/Dream-VLA-7B revision0cd0e25c1771c5b468c74585b5cf35c4a4aa315c;
all23 files verified. Existing official RLDS at
`/data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/datasets/rlds`:
all104 source files verified against pinned HF hashes. No duplicate download
is needed. Python environment: `envs/dream_vla_official`, copied independently
from the existing VLA-Adapter environment and updated for Dream dependencies.

Official bug fixes: include flow action/noise modules in optimizer; route the flow
head through DDP.forward so its gradients synchronize across GPUs; construct
cart reweighting matrix only when needed, at actual sequence length. Additional
user-requested changes: mixed dataset selection, checkpoint retention, complete
training-state restore, safe pause, and completed-update accounting. Dataset
randomness must be checkpointable without changing augmentation distributions.
No separate custom training entrypoint is used; its train.py and config were deleted.

## Resume and pause

The official trainer accepts `--resume_checkpoint PATH` and rejects its old
weights-only resume option. Full checkpoints carry model trainable state,
optimizer, scheduler, per-rank RNG and data iterator state, update count and
configuration identity. Strict resume requires the same world size/batch and
matching source/data configuration. GPU-count changes are not exact resume.
Real two-GPU interruption tests passed functional state restoration. GPU updates
are not bitwise deterministic: parameter relative L2 difference was 3.64e-6,
comparable to 3.67e-6 between two independent restores of the same checkpoint.
Loss, RNG, scheduler, action-head parameters, buffers and progress matched exactly.

Request a safe pause with `scripts/request_dream_pause.py --run-dir RUN`.
Wait for save completion and process exit before reusing GPUs. Do not use
SIGTERM as the normal pause mechanism. `--stop_after_updates` is for smoke
verification without changing the experiment's max-step contract.

## Preserved historical runs

Eight-GPU single-suite run p4hhy9c2 was stopped without a checkpoint after the
user accepted restarting to free GPUs6-7. Six-GPU single-suite run lz3n1xc3 was
subsequently stopped to correct the experiment to mixed training. Its5000-step
checkpoint remains under `experiments/dream_vla_official_6gpu`; it is used only
to test loading/evaluation interfaces, not as initialization for the mixed run.
That checkpoint also predates the flow-head DDP synchronization fix, so its
smoke evaluation must not be used as the corrected Dream baseline score.

# Action-only integration

Implemented as an opt-in path; joint training defaults remain unchanged.

## Contract

- `task=libero_action_only`, `model.action_only=true`, `future_size=0`,
  `image_predict_weight=0`.
- Frozen V-JEPA encoder. Language embeddings and current proprio condition
  the policy through existing cross-attention builders.
- Current visual tokens still pass through the existing visual conditioning
  blocks. They are not a future predictor: there are no future tokens,
  image decoder parameters, image scheduler, or future-latent loss.
- Training uses the existing action flow-matching objective and padding mask.
- Dataset requests only current/past camera frames, while retaining the
  action horizon. Cache lookup selects only the current latent.
- The training API rejects `z_future` instead of silently ignoring it.
- `image_loss=0` is a compatibility placeholder for the upstream trainer's
  logging schema, not a computed visual loss.
- Action-only policy checkpoint loading validates every non-backbone key.
  Full optimizer/scheduler/RNG resume remains the upstream trainer's
  Accelerate state-directory mechanism; policy `.pt` exports are weights only.

## CPU verification

`scripts/test_action_only.py` passed on the server:

- Two optimizer steps with synthetic cached latents and language.
- No encoder gradients; text-projection gradients present.
- No image decoder parameters or image scheduler.
- Future-latent input rejected.
- Current-pixel training path finite.
- Cached and noncached action generation agree.
- Strict model/optimizer serialization roundtrip and policy weights loading.
- Hydra composition of `task=libero_action_only`.

This is not a real LIBERO training or rollout test. CUDA graph paths and real
encoder checkpoint integration have not yet been validated for this variant.

Server test interpreter: `envs/action_only_test/bin/python`, with read-only
PYTHONPATH access to the existing pretraining site-packages. Only Hydra and
OmegaConf dependencies were installed in the separate test environment;
the running pretraining environment was not changed.

## Configuration

The config deliberately requires explicit encoder tag, checkpoint, latent
cache and normalization-stat paths. Original and embodied encoders must
never share latent caches or latent normalization statistics.

Example after preparing and validating those assets:

```bash
python scripts/train.py task=libero_action_only \
  encoder_tag=embodied_030000 \
  model.backbone.ckpt_path=/absolute/path/embodied_vjepa_encoder_030000.pt \
  data.train.rgb_latent_cache_dir=/absolute/path/embodied_latents \
  model.latent_norm_stats_path=/absolute/path/embodied_latent_norm_stats.json
```

Training data paths and simulator version must be selected explicitly before
running this command. The inherited default dataset paths are placeholders
for the upstream MuJoCo 3.3.2 dataset, not the discovered local legacy data.

## Existing server assets

- Original HDF5 demonstrations:
  `/data/users/jiachangzhang/DNFT4RP/upstream/libero/libero/datasets/`.
  Four suites, 10 HDF5 task files each. Episode/content completeness not audited.
- LIBERO source and assets:
  `/data/users/jiachangzhang/DNFT4RP/upstream/libero/`.
- Working physics environment:
  `/data/users/jiachangzhang/RTC/.venv-libero/`
  (Python 3.9, MuJoCo 2.3.7, robosuite 1.4.0, torch 1.13.1).
  A real spatial-task reset, initial-state restore and action step passed
  using `MUJOCO_GL=glfw`, no camera rendering, CUDA hidden.
- The DNFT4RP `.venv-libero` is not sufficient as-is: torch and MuJoCo were
  not installed there. Do not confuse it with the working RTC environment.
- Candidate public LeRobot data:
  `/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA/`.
  Spatial metadata reports v2.1, 432 episodes, 20fps, two 256px AV1 cameras,
  7D action and 8D state. Do not assume this matches upstream MuJoCo 3.3.2
  recollected data. Do not modify shared files.
- RoboTwin source/assets:
  `/data/users/jiachangzhang/RoboTwin/`, with
  `/data/users/jiachangzhang/.venv-robotwin/`. Not smoke-tested in this task.
- LIBERO-Plus evaluation wrapper:
  `/data/public/bohantan/starVLA/examples/simBenchmarks/LIBERO-plus/`.
  A complete simulator/assets installation has not been verified.

CPU OSMesa rendering failed because the GL library was unavailable. Physics
passed without rendering. RGB rendering, simulator-policy bridge, actual
LeRobot batch loading, language cache availability, encoder-specific latent
caches and full training dependencies remain to be validated. No downstream
training run was launched and no pretraining GPU was used.

## Follow-up verification (2026-10-03)

`scripts/check_real_encoder.py` passed on CPU with the public spatial dataset:
two real AV1 camera streams, causal frame indices [0, 2, 4, 6], tiled 224x448.
Both the original checkpoint and the embodied step-20000 EMA export loaded
strictly through the downstream VJEPABackbone and returned finite frozen
features of shape [1, 392, 1024]. This checks compatibility, not representation
quality, temporal-recipe equivalence, or a training result.

`scripts/check_real_dataset.py` passed using the actual BaseLerobotDataset in
cached mode (no RGB decode): 52,970 frames, indices 0/6/52969 checked, actions
[16,7], states [17,8], one current image position and correct end-of-episode
padding (15 padded actions in the last sample). Real task strings were loaded.
No normalization processor, text embeddings or precomputed RGB latents were
substituted with fake data in this test; these stages were not executed.

Reports are under `/data/users/jiachangzhang/eb_vjepa/experiments/action_only/`:
`real_encoder_check.json` and `real_dataset_check.json`.
The separate test environment gained datasets/accelerate and small utility
dependencies; the pretraining environment remained unchanged. It is still a
test overlay, not a validated production training environment.

Bounded searches of the user's directories and selected public directories
did not locate reusable umT5 weights/text caches. This is not an exhaustive
claim that none exist on the server. Text encoding, per-encoder cache generation,
full trainer resume and rendered closed-loop evaluation remain outstanding.

## CPU simulator compatibility update

The previous OSMesa limitation is resolved using a project-local extracted
libosmesa6 package. `scripts/run_cpu_libero_smoke.sh` in the project root now
passes in the Python 3.11 action_only_test overlay with torch 2.11, MuJoCo 2.3.7
and robosuite 1.4.0. CUDA is hidden. Real LIBERO task initialization, three
zero-action steps, both RGB cameras and finite robot states were verified.
Report: `experiments/action_only/unified_rgb_smoke.json` on the server.
This does not validate policy inference, action denormalization or task success.

Native PyAV decoding is implemented in the vendored video_utils.py; the real
AV1 regression test compared RGB frames exactly against sequential decoding,
including repeated/out-of-order requests. Full policy data processing and
per-encoder cache identity enforcement remain separate acceptance gates.

## Encoder-specific cache guards

The action-only configuration now supplies encoder_checkpoint_path to the
dataset. Precomputation stores the checkpoint SHA-256 in every episode and
suite metadata, and a preprocessing-recipe identity sidecar. A different
checkpoint or recipe cannot reuse that output directory. Episode files and
completed suite metadata are published by atomic rename.

Action-only loading checks suite and episode identities against the configured
checkpoint and checks clip length/frame interval. Latent statistics require
consistent encoder identities, use only episodes listed in completed metadata,
and propagate the identity into the statistics file. Policy construction checks
those statistics against its backbone checkpoint and rejects nonfinite or
nonpositive normalization scales.

CPU `scripts/test_encoder_identity.py` passed: known mean/std, rejection of
missing/mixed identities, changed checkpoint rejection, and ignoring unlisted
stale episode files. Fixtures are unit-test data only, not substitute training
data or encoders. `scripts/test_action_only.py` also passed again.
Real final-encoder cache generation and full training acceptance remain pending.

## Real action normalization check

`scripts/check_action_normalization.py` passed on the existing public spatial
dataset (432 episodes, 52,970 frames). It reads all action/state parquet rows
for min/max and checks the actual processor on first/interior/final samples.
Action/state inverse-normalization error was at most 5.96e-8; the final sample
has 15 padded actions, with padded motion dimensions zeroed as expected.
Gripper labels are exactly {0,1}. The saved single-suite statistics are diagnostic
only, not the selected four-suite production statistics.

`scripts/check_libero_conventions.py` matched converted episode 0 to the original
spatial HDF5 demo_40 using all six motion-action dimensions and strictly ordered
frame matches (max error 2.90e-8). Gripper labels satisfy converted=(1-raw)/2
exactly, supporting the existing simulation conversion on this episode.
However, matched observation states differ (eef max error 0.04066; gripper
state max error 0.00884). Offsets from -2 through +2 frames do not eliminate
the difference. Dataset state provenance/observation alignment is unresolved;
do not treat the modality metadata or this one action match as full validation.
No data or simulator-state transformation was changed to force agreement.

## Observation replay follow-up

The official OpenVLA regeneration code replays actions after ten settling steps,
removes no-ops, and records pre-action observations, rather than copying original
HDF5 observations. Source:
https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py

`check_libero_conventions.py --replay` reproduced this protocol for the matched
110-frame demo on CPU. Mean state error vs parquet dropped to 2.077e-5; maximum
error was 0.001041 (gripper), with position errors below 8.60e-5. This supports
regeneration as the reason for the earlier large discrepancy but is not exact
reproduction: matches_at_1e_5=false. No observation values were modified.
Report: experiments/action_only/libero_replay_conventions_check.json.
Still only one demonstration; image orientation and policy rollout acceptance
remain separate checks.

## Camera orientation and policy identity follow-up

`check_libero_conventions.py --replay --check-images` compared each camera's
first real AV1 frame to the CPU-replayed observation. Among unchanged, 180-degree
rotation and the two single-axis flips, rotation was best for both cameras.
Mean absolute RGB errors (0..255) after rotation: 14.47 agent / 14.95 wrist;
unchanged: 68.46 / 79.01. This supports the existing evaluator's 180-degree
rotation but is not pixel-exact reconstruction or a full image-domain audit.
Report: experiments/action_only/libero_camera_conventions_check.json.

Runtime now stores the encoder SHA-256 in policy checkpoint backbone metadata.
Action-only policy loading and the evaluation config check reject a missing or
mismatched encoder identity. The unit action-only test passed again, including
a negative test that changes an identity fixture after saving the policy.
No real trainer/resume or trained-policy rollout result is claimed.

## Resumable training-state safeguards

Trainer state now records a resume contract: model/data config, encoder and
normalization hashes, world size, batch/accumulation, seed, precision, schedule
and total steps. Action-only full-state resume rejects missing progress or a
different contract before Accelerate loads anything.

Checkpoint state writes go to state/pending first. After successful save and
trainer metadata write, the old latest is moved to previous and pending becomes
latest. A write failure preserves the existing latest. An interruption during
the final directory renames may leave the old complete state at previous;
recovery then requires explicitly selecting that directory. This is not a
claim of a fully atomic multi-directory transaction.

`scripts/test_trainer_state.py` passed on CPU using actual Accelerate state
serialization with a small explicitly unit-test-only linear model. It verifies
parameter, optimizer, scheduler, Torch RNG and sampler offset restoration,
contract/incomplete-state rejection, and injected write failure preservation.
This is checkpoint-plumbing coverage, not real V-JEPA/LIBERO training acceptance.

## Pro assets and actual language routing

Official Pro snapshot download completed (675 files fetched); 353 entries in
upstream SHA256SUMS.txt were checked. Assets live in the isolated
checkpoints/benchmark_assets/libero_pro directory; config is
experiments/action_only/pro_config/config.yaml. The official Pro source,
libero_spatial_lan task 0, passed CPU reset/init-state/three-step simulation
with both RGB cameras. Report: experiments/action_only/pro_rgb_smoke.json.
This is one simulator task, not full Pro or policy evaluation.

Found and fixed a prompt-routing issue: task.language derives the original
sentence from the filename, whereas the Pro BDDL contains the paraphrase.
get_libero_env now gets its instruction through the official BDDL parser.
test_libero_language.py checked all ten spatial-language tasks; all ten now
return the actual perturbed BDDL language. Report: pro_language_check.json.
This validates prompt extraction only, not umT5 encoding or learned behavior.

## Action-only DDP correction

The inherited trainer called training_loss on an unwrapped model instead of
the DDP forward lifecycle. Action-only training now calls self.model(batch),
with FlowPolicy.forward dispatching to training_loss. Its Accelerate DDP kwargs
enable find_unused_parameters and gradient_as_bucket_view. Joint-mode behavior
is unchanged.

Two CPU/Gloo ranks ran test_action_only_ddp.py for three optimizer steps with
two-microstep accumulation and no_sync on the first microstep: PASS. Gradients
for all 92 participating parameter tensors matched across ranks; 14 tensors
in the final image block had no gradient, as expected without future prediction.
These counts describe the tiny test model, not the production architecture.
The action-only and trainer-state unit regressions also passed. This validates
distributed plumbing only; real GPU/LIBERO/text-conditioned acceptance is pending.

## Downstream review fixes: 2026-10-03

- Evaluator accepts the confirmed Pro lan/object/swap/task suffixes for the four
  base suites and inherits each base suite's rollout limit. Unknown suites fail.
- Action-only data reads fail on the requested sample instead of substituting
  another index, in both RobotVideoDataset and BaseLerobotDataset.
  Joint-mode retry behavior is unchanged.
- umT5 state loading is strict; missing/unexpected keys and shape mismatches stop
  loading rather than leaving uninitialized parameters.
- Online W&B identity (ID, project, workspace) is persisted in trainer state
  and wandb_run.json. Full-state resume restores the same ID with resume=must;
  mismatched projects and missing checkpoint identity are rejected. This is
  not a claim of online resume support in W&B offline mode.

Server CPU regressions passed: test_downstream_guards.py (five tests),
test_trainer_state.py (including W&B identity restoration), and
test_action_only.py. W&B initialization in the guard test uses a mock; strict
text loading uses a tiny explicitly unit-only model. Real umT5 loading,
final-encoder training/resume, and learned-policy closed-loop acceptance remain
required before releasing the launch gate.

## Real acceptance update: 2026-10-03

Final 30k EMA strict load and exact equality with final.pt passed (302 finite
tensors). Real umT5 strict loading and all 40 real LIBERO prompt embeddings
passed. Actual real-episode single-GPU training/resume, two-GPU NCCL/DDP
training/resume, and batch-32 single-GPU training/resume passed. W&B resumed
the same online ID in each test. Actual 3-predicted-action closed-loop plumbing
passed in clean LIBERO, Plus, and Pro spatial-language with the 8-step-trained
policy. These are acceptance checks, not reported benchmark success rates.

Full final-encoder cache preparation subsequently stopped on corrupt
LIBERO-Goal wrist video episode_000082.mp4: PyAV/libdav1d and independent
libaom-av1 fail; file SHA256 matches its downloaded-source metadata.
No episode was skipped or replaced. Production cache and formal train launch
remain blocked pending an explicit provenance-preserving data repair decision.
Details, commands, source hash, logs, and W&B IDs are in experiments/action_only/RUN.md.

# Automatic downstream handoff

## Current Priority: Full DROID 72h Pretraining

2026-10-03: action-only production status is complete, returncode 0.
The user authorized an independent full-DROID run from official weights.
`experiments/ctp_full_72h` now owns all eight GPUs through acceptance and
formal training. Check its status and live processes before downstream work.
Do not start competing evaluation, cache extraction or training jobs.
This supersedes earlier GPU-availability/start instructions below.

Requested on 2026-10-03 (Asia/Shanghai): final CTP EMA encoder, LIBERO action-only
policy training, then closed-loop LIBERO / LIBERO-Plus / LIBERO-PRO evaluation.
No downstream success or generalization result exists yet.

## Active waiting mechanisms

- Server watcher PID: 1726443. Started 2026-10-03 01:54:51 Beijing.
- Script: `/data/users/jiachangzhang/eb_vjepa/scripts/watch_action_pipeline.py`.
- Log/status: `experiments/action_only/watcher.log`, `watcher_status.json` on server.
- Independent of desktop app; polls every 120 seconds, expires after seven days.
- App heartbeat: `embodied-v-jepa`, every 30 minutes, continues environment/code
  preparation in this thread. This is separate from the server launcher.
- Current phase: waiting for CTP final AND a validated `ready_plan.json`.
  The ready plan does NOT exist yet. Automatic training is NOT armed yet.

## Launch gates

The watcher acquires an exclusive lock and refuses to rerun an already executed
pipeline. It requires hashed validation reports, a concrete Pro identity, all
five stages, CTP process exit, sane final checkpoint with EMA weights, final EMA
export, and specified free GPU memory. Commands run sequentially with separate
logs. A failing command or missing expected artifact stops the sequence.

The audited plan must contain: `validated`, `pro_identity`, `validation_reports`
with path/SHA256, `required_free_mib`, environment variables and ordered stages:
`prepare_final_latents`, `train`, `eval_libero`, `eval_plus`, `eval_pro`.
Each stage supplies absolute executable argv, cwd, optional env and expected
artifacts. `{encoder}` and `{step}` are resolved from the actual final checkpoint.
Do not create this file before real training/resume and evaluation smoke tests.

## Benchmark identity

Use official LIBERO-PRO: https://github.com/Zxy-MLlab/LIBERO-PRO
and LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus .
The previously discovered VLA-Adapter LIBERO-Pro model cache is NOT evidence of
a simulator installation. Record checked-out commits and asset revisions before
evaluation. Never report a partial task subset as the full benchmark.

## Preparation still required

See `RoboActualizer/ACTION_ONLY.md` for completed CPU / real encoder / data tests.
Remaining: text assets and real language embeddings; final-encoder-specific
latents and normalization; validated production dependencies; actual trainer
save/resume; rendered policy rollout; Plus/Pro assets and adapters; fixed eval
task/seed protocol. Download preparation is CPU/network only. Do not take GPUs
from CTP to train another model. The currently tested 20k encoder is NOT the
selected final encoder for automatic downstream training.

Text asset job: `scripts/prepare_text_assets.py`, downloads only the Wan umT5
encoder and tokenizer into `/data/.../eb_vjepa/checkpoints/text`, not the full
video generator. Logs and launch PID are in `experiments/action_only/`.

## Watcher command

```bash
env TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp \
 /data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/python \
 /data/users/jiachangzhang/eb_vjepa/scripts/watch_action_pipeline.py \
 --root /data/users/jiachangzhang/eb_vjepa --detach
```

Do not run a second copy. Use the status file and lock to identify the active one.

## Environment progress

- LIBERO CPU RGB smoke passed after extracting the Ubuntu `libosmesa6`
  package into `cache/osmesa/root`, without sudo or system package changes.
  RTC simulator environment: one task reset, init-state restore, action step,
  RGB [64,64,3], pixel standard deviation 50.44459. This is not a trained-policy
  rollout or success-rate measurement.
- Use `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, and prepend
  `/data/users/jiachangzhang/eb_vjepa/cache/osmesa/root/usr/lib/x86_64-linux-gnu`
  to `LD_LIBRARY_PATH`. Set `MESA_SHADER_CACHE_DIR` to the project cache, since
  the home/system filesystem is full.
- Official Plus checkout: `repos/LIBERO-plus-official`, commit
  `4976dc30028e805ff8094b55501d532c48fec182`. Extra assets are NOT installed yet.
- Official Pro checkout preparation: `repos/LIBERO-PRO-official`; commit and
  required BDDL/init files must be checked after clone completes.
- Text download job currently uses ModelScope. Observed initial speed about
  0.21 MB/s for an 11.4 GB file; readiness by morning is not guaranteed.
  Do not mistake `.incomplete` files for completed weights.

## Heartbeat verification, 2026-10-03 02:45 Beijing

- CTP was still active; sampled optimizer step 22005, loss 0.540272,
  throughput 56.085 clips/s. No competing GPU workload was started.
- The separate Python 3.11 action_only_test environment now passes a real
  LIBERO spatial-task CPU smoke with MuJoCo 2.3.7, robosuite 1.4.0 and
  PyTorch 2.11.0+cu128. Both RGB cameras are nonblank [64,64,3]; end-effector
  position/quaternion and gripper states are finite; three zero-action steps pass.
  This is an environment smoke, NOT a policy rollout or success-rate result.
- Reproduce on server:
  `bash /data/users/jiachangzhang/eb_vjepa/scripts/run_cpu_libero_smoke.sh`
- Report: `experiments/action_only/unified_rgb_smoke.json`.
  RTC environment separately passed the same camera/state checks; report:
  `experiments/action_only/rtc_rgb_smoke.json`.
- Isolated dependencies added: llvmlite 0.50.0 (matching numba 0.68), future,
  matplotlib and plotting dependencies. Pretraining environment unchanged.
  The smoke explicitly loads trusted local LIBERO numpy initial-state files
  with weights_only=False, instead of changing torch.load globally or editing
  the shared LIBERO source. Its launcher hides CUDA and limits OpenMP to 2.
- Official Pro checkout completed: `repos/LIBERO-PRO-official`, revision
  `eafdb809426b13153aa1e4c42d6601844217dfec`. Required upstream BDDL/init
  assets: https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro .
- Text weights still downloading (about 5% at the last sample, roughly
  0.21 MB/s). Real text encoding and full trainer/resume remain unverified.
  Final latent cache provenance checks and Plus/Pro assets/adapters are also
  outstanding. ready_plan.json is still absent; automatic training is NOT armed.

## Heartbeat: encoder cache identity guards

CTP sampled at step 22317, still active, with watcher waiting and no readiness
plan. Text download was about 6%; no GPU preparation job launched.

Implemented checkpoint SHA-256 propagation through RGB cache generation,
per-episode loading and latent normalization statistics. Action-only dataset
and policy reject mismatched identities. Cache output also records the
preprocessing recipe; writes use atomic episode/metadata publication. Existing
unidentified caches are deliberately rejected instead of relabeled.

Verification on CPU (CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=2):

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
 TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp \
 PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/src:/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages \
 /data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python \
 /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/test_encoder_identity.py
```

PASS; output saved to `experiments/action_only/encoder_identity_unit.log`.
The same environment running `scripts/test_action_only.py` also passed.
Precompute and dataset source compile checks passed. These are code/unit checks,
not real final-encoder cache or trainer/resume acceptance. Remaining gates:
text weights/encoding, real cache generation, action normalization and simulator
action conventions, actual trainer/resume, policy rollout, Plus/Pro assets and
adapters. Automatic training remains unarmed until those checks pass.

## Heartbeat: real action normalization and conventions

CTP sampled at optimizer step 23050, loss 0.540694, 51.40 clips/s; final not
present and watcher waiting. Text download about 9%. No GPU work was added.

New CPU scripts in RoboActualizer/scripts:

- check_action_normalization.py: PASS on all 432 spatial episode parquet files
  for min/max statistics, with processor roundtrips at indices 0/6/52969.
  Max action/state error <=5.96e-8; 15 final padded actions handled correctly.
  Report: action_normalization_check.json. Its diagnostic_stats file is NOT
  four-suite production normalization.
- check_libero_conventions.py: converted episode 0 matches HDF5 demo_40 motion
  actions (max error 2.90e-8), ordered unique matches, converted gripper=(1-raw)/2
  exactly. Report: libero_conventions_check.json. State values differ from the
  matching HDF5 observations and simple temporal offsets do not fully explain
  this. Observation provenance/alignment remains a launch gate, not a passed test.

Exact server commands, after setting the following environment:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp
export HF_HOME=/data/users/jiachangzhang/eb_vjepa/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/src:/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages
ROOT=/data/users/jiachangzhang/eb_vjepa
DATA=/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot
$ROOT/envs/action_only_test/bin/python $ROOT/repos/RoboActualizer/scripts/check_action_normalization.py --dataset $DATA --output $ROOT/experiments/action_only/action_normalization_check.json
$ROOT/envs/action_only_test/bin/python $ROOT/repos/RoboActualizer/scripts/check_libero_conventions.py --dataset $DATA --hdf5-suite /data/users/jiachangzhang/DNFT4RP/upstream/libero/libero/datasets/libero_spatial --output $ROOT/experiments/action_only/libero_conventions_check.json
```

The first normalization invocation failed because HF_DATASETS_CACHE was not
redirected from the full home filesystem. The recorded command fixes that;
the rerun passed. Text download, real training/resume, final caches and full
closed-loop benchmark acceptance are still pending. ready_plan remains absent.

## Replay follow-up and Plus asset download

Latest CTP sample: step 26022, loss 0.553573, 56.07 clips/s. Main CTP and the
waiting watcher remain untouched. Text weights about 24%, still incomplete.
The earlier RUN synchronization succeeded on retry.

The CPU replay diagnostic completed successfully as a process, but did NOT
meet its exact-state threshold. Following official OpenVLA regeneration
(10 settling steps, no-op removal, pre-action observations), the matched
110-frame trajectory had mean state error 2.077e-5 and max 0.001041 vs parquet.
This explains much of the original-state discrepancy without modifying data.
It does not establish exact full-dataset provenance or a trained-policy score.
Source: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py
Report: `libero_replay_conventions_check.json`.
Reproduce the preceding check_libero_conventions.py command with --replay,
using the OSMesa / LIBERO_CONFIG_PATH / NUMBA_CACHE_DIR / MPLCONFIGDIR settings
from scripts/run_cpu_libero_smoke.sh and adding the own LIBERO source directory
to PYTHONPATH. The existing HDF5 and parquet remain read-only.

Plus asset download is now a separate CPU/network-only process, PID 2619569.
Script: scripts/prepare_plus_assets.py; exclusive flock, resume, atomic status,
fixed revision, official SHA-256 verification, no automatic extraction or GPU work.
Official archive size: 6,395,849,578 bytes; source dataset Sylvest/LIBERO-plus.
Log/status: plus_assets_download.log / plus_assets_download.json under action_only.
Current phase: downloading, NOT verified/installed. Do not start another copy.

Launch command:

```bash
env CUDA_VISIBLE_DEVICES='' HTTPS_PROXY=http://127.0.0.1:7896 \
 TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp \
 PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages \
 /data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python \
 /data/users/jiachangzhang/eb_vjepa/scripts/prepare_plus_assets.py \
 --root /data/users/jiachangzhang/eb_vjepa --detach
```

Pro source includes assets and some BDDL/init directories, but this does not
prove the complete official perturbation suite assets are available. Pro
environment/adapters, text encoding, final latent caches, real trainer/resume
and policy closed-loop tests remain required. No ready_plan was created.

## Heartbeat: camera orientation and download recovery

CTP sample: step 26776, 58.42 clips/s; text download about 27%.
Plus downloader PID 2619569 failed with curl exit 92 (HTTP/2 transfer error).
The downloader now selects HTTP/1.1, with five bounded retries and ten-second
retry delay. Relaunched with the previously recorded command; PID 3342852,
exclusive lock retained, archive resumed successfully and bytes were increasing.
State remains downloading, not installed. No competing GPU run exists.

CPU check_libero_conventions.py with --replay --check-images completed for the
same real episode, using the previously recorded OSMesa/PYTHONPATH environment.
Output: experiments/action_only/libero_camera_conventions_check.json.
Both cameras select rotate_180 as the lowest-error orientation. MAE after
rotation is 14.47/14.95 on [0,255], versus 68.46/79.01 unchanged. This supports
orientation only; appearance is not pixel-exact and broader validation remains.

Added encoder SHA-256 metadata to policy construction and fail-closed checks in
action-only load_checkpoint and evaluation checkpoint/config matching. Ran the
previously recorded CPU test_action_only.py command: PASS, now also including
wrong-encoder rejection using an explicitly synthetic identity-only unit fixture.
The fixture is never used by actual training or evaluation.

Automatic training remains unarmed. Downloads, real text encoding, final EMA
caches, actual trainer/resume and learned-policy clean/Plus/Pro rollouts are not
yet complete. All changes in this heartbeat are downstream-only.

## Heartbeat: full-state resume plumbing

CTP sample: step 27542, 57.05 clips/s; text weights about 30%; Plus download
status downloading. Watcher still waiting with no final or validated plan.

Downstream trainer changes: action-only full-state resume now validates model,
dataset, encoder/statistics identities and training configuration before loading
Accelerate state. Missing epoch/batch progress is an error rather than silently
restarting the dataloader. New state is written into pending before rotating
latest to previous; a failed write retains the old latest. previous is retained
for explicit recovery if interrupted between the final directory renames.

CPU verification command:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
 TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp \
 PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/src:/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages \
 /data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python \
 /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/test_trainer_state.py
```

Result: trainer_state_unit=PASS, real_pipeline_acceptance=false. Actual
Accelerate save/load restored parameters, optimizer, scheduler, Torch RNG and
sampler progress in a small unit-only model; incompatible/incomplete resumes
were rejected. Injected save failure preserved latest. The test's first version
incorrectly accessed AcceleratedScheduler.last_epoch; corrected to its wrapped
scheduler and reran successfully. Real text-conditioned V-JEPA/LIBERO trainer
smoke/resume is still required, along with final caches and benchmark rollouts.
No GPU task or readiness plan was started by this heartbeat.

## Heartbeat: benchmark asset preparation

CTP sampled at step 28293, 57.50 clips/s. Text weights about 34%; CTP final
still absent. Plus PID 3342852 failed with curl 56 (TLS unexpected EOF).
Updated downloader to retry only transient transport codes using a fresh curl
process/offset each attempt, bounded to 100 attempts or 48 hours between checks.
Each transfer has a 30-minute timeout; HTTP/range errors stop instead of retrying.
Relaunched using the existing command as PID 521632; confirmed resuming from
120,970,055 bytes. Hash/size verification remains mandatory.

Pro official BDDL/init download started, PID 2057705, 675 files at a fixed
revision of zhouxueyang/LIBERO-Pro. Target is separate from the source checkout:
checkpoints/benchmark_assets/libero_pro. It uses two network workers, exclusive
flock, status/PID records and the upstream SHA256SUMS.txt. No GPU or shared
LIBERO config is changed. Status/log: pro_assets_download.json/.log.

Exact launch command:

```bash
env CUDA_VISIBLE_DEVICES='' HTTPS_PROXY=http://127.0.0.1:7896 \
 HF_HOME=/data/users/jiachangzhang/eb_vjepa/cache/huggingface \
 TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp \
 PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages \
 /data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python \
 /data/users/jiachangzhang/eb_vjepa/scripts/prepare_pro_assets.py \
 --root /data/users/jiachangzhang/eb_vjepa --detach
```

Prepared isolated Pro config at experiments/action_only/pro_config/config.yaml.
The CPU smoke helper accepts --suite and reports the actual benchmark module
path. The launcher supports explicit LIBERO_SOURCE and LIBERO_CONFIG_PATH so
clean and Pro run in separate processes without import ambiguity.
After asset verification, the targeted Pro language-suite smoke command is:

```bash
env LIBERO_SOURCE=/data/users/jiachangzhang/eb_vjepa/repos/LIBERO-PRO-official \
 LIBERO_CONFIG_PATH=/data/users/jiachangzhang/eb_vjepa/experiments/action_only/pro_config \
 bash /data/users/jiachangzhang/eb_vjepa/scripts/run_cpu_libero_smoke.sh \
 --suite libero_spatial_lan \
 --output /data/users/jiachangzhang/eb_vjepa/experiments/action_only/pro_rgb_smoke.json
```

Prepared does not mean passed: the Pro smoke above still needs execution and
is only one task, not the full Pro evaluation. Automatic training remains unarmed.

Pro follow-up result: download_verified, 353 official checksum entries checked.
The Pro smoke command above ran and PASSED on the official module; both cameras
were nonblank and robot states finite. Report pro_rgb_smoke.json records source
module/config identity. No policy was involved.

Found a language perturbation bug in the shared evaluator: it used the original
filename-derived task.language. It now uses the actual BDDL language through
the official parser. Verified ten Pro spatial-language tasks; all ten yield
their perturbed instructions. Report pro_language_check.json, status PASS.
Exact test command:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
 TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp \
 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
 LD_LIBRARY_PATH=/data/users/jiachangzhang/eb_vjepa/cache/osmesa/root/usr/lib/x86_64-linux-gnu \
 LIBERO_CONFIG_PATH=/data/users/jiachangzhang/eb_vjepa/experiments/action_only/pro_config \
 NUMBA_CACHE_DIR=/data/users/jiachangzhang/eb_vjepa/cache/numba \
 MPLCONFIGDIR=/data/users/jiachangzhang/eb_vjepa/cache/matplotlib \
 PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/repos/LIBERO-PRO-official:/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages \
 /data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python \
 /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/test_libero_language.py \
 --output /data/users/jiachangzhang/eb_vjepa/experiments/action_only/pro_language_check.json
```

The first prompt-test invocation omitted TMPDIR and failed on the full system
filesystem; the command above fixes that and passed. Text embeddings, real
training/resume, final encoder latents, Plus installation and learned-policy
closed-loop evaluation remain pending; no validated readiness plan exists.

## Heartbeat: action-only DDP lifecycle

CTP sample: step 29052, 57.59 clips/s, about 950 steps remaining to its 30k cap.
Text download about 37%; Plus downloading/resuming (attempt 4, 458 MB recorded).
No CTP code/config/process was changed.

Found and fixed action-only training bypassing DDP.forward when calling
training_loss. It now enters the prepared model's forward, and Accelerate DDP
uses find_unused_parameters=True and gradient_as_bucket_view=True for this mode.
The tiny action-only graph has genuinely unused final-image-block parameters;
no artificial loss or future-prediction branch was added to force gradients.

Exact distributed CPU test command:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
 TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp \
 PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/src:/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages \
 /data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python \
 -m torch.distributed.run --standalone --nproc_per_node=2 \
 /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/test_action_only_ddp.py
```

PASS: two Gloo ranks, three optimizer steps, accumulation=2, first microstep
no_sync, 92 participating parameter tensors with matching reduced gradients,
14 unused final-image-block tensors in the unit model. Test explicitly reports
real_training_acceptance=false. Re-ran test_action_only.py and
test_trainer_state.py in the recorded CPU environment: both PASS.
These synthetic unit fixtures are not used by production training/evaluation.
The actual eight-GPU pretraining remained the only GPU workload launched here.
Real downstream training/resume and benchmark rollout gates still remain.

## CTP curve audit: snapshot through step 29683

Read-only inspection of the running CTP metrics and upstream
`Lambda_LinearWarmupHold` confirmed:

- Total loss is mean(prediction) + lambda * mean(context), across two masks.
- Lambda is zero through optimizer step 15001, then ramps toward 0.5 at
  step 30001 (the forward receives step - 1). No schedule was changed.
- Latest snapshot: prediction mean 0.4462621361, context mean 0.2531977221,
  lambda 0.4894, total 0.5701771379. Reconstructed total differs only by
  floating-point rounding; maximum error over all 29683 records was 1.35e-7.
- Steps 14000-14999 prediction means: 0.49363 / 0.52854; steps 29000-29683:
  0.42261 / 0.47437. Raw prediction and context components decreased late,
  despite the increasing weight raising total loss.
- Maximum gradient norm was 11.6083 at step 25722; adjacent steps 25721 and
  25723 were 0.25195 and 0.20325. Loss was 0.54861 / 0.56259 / 0.56065.
  No nonfinite values in the checked loss, gradient and component records.
- The trainer's infinite clip threshold measures norms but does not clip
  finite gradients. `encoder_grad_norm` measures patch_embed.proj.weight,
  not all encoder parameters. EMA max delta also covers only a probe.

This explains the 15k context turn and the rising total; it does not establish
representation quality, convergence, or the cause of the isolated spike.
Do not compare total loss across stages as a fixed objective.
Local evidence: experiments/ctp_metrics_snapshot.jsonl (read-only SCP snapshot).

## CPU-only real latent-cache smoke

Added explicit CPU/device selection to precompute_rgb_latents.py, resolved
Hydra configuration resolvers, and load only the frozen backbone rather than
constructing a random action policy to obtain its backbone. GPU remains the
default and missing CUDA fails explicitly.

Command: bash /data/users/jiachangzhang/eb_vjepa/scripts/run_real_cache_smoke.sh

The bounded smoke uses the real first LIBERO spatial episode and the strict
20k EMA export, with two CPU threads and batch size two. Output directory:
experiments/action_only/cache_smoke_020000. This is diagnostic only, not the
final-encoder production cache or a downstream training acceptance result.
At this entry the encode process is still running; completion is not claimed.

## Completion and readiness check: 2026-10-03

CTP reached optimizer step 30000 and 3840000 clips. final.pt (4.8G) and
embodied_vjepa_encoder_030000.pt (1.2G) were written at 07:59 server time.
All eight GPUs subsequently reported 0% utilization and 5-6 MiB used.
The final export's strict load validation is still pending.

The CPU real-cache smoke above exited successfully: one real 110-frame episode,
43120 latent rows; latent_norm_stats.json written with global mean -0.006309
and global std 2.134439. This remains a diagnostic 20k encoder cache only.

All four LIBERO LeRobot suite directories are present. Pro assets download
verified 353 checksum entries. Plus download remains in progress, with
850710100 of 6395849578 bytes recorded at attempt 8. umT5 BF16 weights are
about 4.72G / 11.4G; text readiness remains blocked on this download.
Watcher sees final.pt but no ready_plan.json, so no downstream training started.

Action-only task configuration now explicitly enables online W&B in project
eb-vjepa with run name libero_action_only_${encoder_tag}, replacing the inherited
roboactualizer project for this task only. No live downstream W&B run is claimed.

## Downstream review repair verification: 2026-10-03

Deployed fixes to the server RoboActualizer tree:
1. Pro suite step limits support exactly the confirmed four suffixes
   lan/object/swap/task across spatial/object/goal/10.
2. Action-only read failures cannot fall back to a randomly selected sample
   at either dataset layer. Joint-mode behavior is preserved.
3. umT5 uses strict state loading.
4. Online W&B ID/project/workspace are saved in trainer checkpoint metadata.
   Resume initializes W&B only after restoring state, with the same ID and
   resume=must. Missing identity or a changed project/workspace fails.

Exact regression commands (server shell):

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp
export PYTHONPATH=/data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/src:/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/lib/python3.11/site-packages
/data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/test_downstream_guards.py -v
/data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/test_trainer_state.py
/data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/test_action_only.py
```

All exited 0. Guards: 5/5 PASS. Trainer state: PASS with W&B identity roundtrip.
Action-only: PASS for backward, frozen encoder, language gradients, no future
inputs/head, pixel path, cached inference parity and state/optimizer roundtrip.
The guards use mocked W&B calls and a tiny text-loader unit fixture, not
production text weights. No real pipeline acceptance or benchmark score is
implied. Full real-data training/resume and learned-policy rollout remain gated.

## Real acceptance: 2026-10-03, 09:30-10:10 server time

Confirmed using the real final 30k EMA encoder and real LIBERO data:

- Final export strictly loads through the downstream backbone. All 302 exported
  tensors are finite and exactly equal to final.pt's ema_encoder.
  SHA256: b920571c1aa6446b646b9353ef131261b3ce336eb709057bc4f00c40dca4f543.
  Evidence: final_encoder_validation.json.
- One complete spatial episode encoded into cache_smoke_final, with latent
  normalization statistics. This diagnostic subset is not the production cache.
- Real umT5 strictly loaded; all 40 LIBERO task prompts encoded into text_cache.
- Real single-GPU training ran 8 optimizer steps and resumed from step 4 to 8.
  W&B online ID 32e197da in eb-vjepa. Two-GPU NCCL/DDP repeated this successfully
  on GPUs 5,6, W&B ID 1b787c76. Batch 32 single-GPU training/resume also passed
  on GPU 7, W&B ID aaf5e3bb. All are named smoke/acceptance, not formal runs.
- In the DDP run, step-8 action loss was 1.1659 both before and after restoring
  step 4; image loss was zero. Replaying already logged optimizer steps causes
  W&B to ignore those duplicate historical log points, as expected.
- An API incompatibility discovered in actual W&B initialization was fixed:
  use uuid-based IDs rather than unavailable wandb.util.generate_id.
  Avoid duplicate final checkpoint save, so previous retains a real earlier
  state. Updated five guard regressions passed.
- clean_learned_smoke.json, pro_learned_smoke.json, plus_learned_smoke.json:
  each PASS for 3 actual predicted-action environment steps, including real
  text encoding, online EMA visual encoding, action denormalization, and two
  nonblank 256x256 camera observations. Pro uses libero_spatial_lan's actual
  paraphrase. These use an 8-step-trained policy; benchmark_score is null.
  Acceleration caches were disabled for this plumbing check; production
  accelerated inference has not been accepted.
- Plus archive extracted (448799 files). Installed Wand only into the isolated
  test venv; ImageMagick/lcms/lqr/fftw libraries extracted without sudo into
  cache/imagemagick. MAGICK_HOME points to cache/imagemagick/home and its lib
  symlink. Plus source-relative assets symlink points to the extracted assets.
  The smoke now calls the benchmark's official init-state mapping, preserving
  Plus perturbation semantics.

Exact preparation/acceptance commands:

```bash
bash /data/users/jiachangzhang/eb_vjepa/scripts/run_action_acceptance.sh
python3 /data/users/jiachangzhang/eb_vjepa/scripts/extract_plus_assets.py
bash /data/users/jiachangzhang/eb_vjepa/scripts/install_plus_native.sh
bash /data/users/jiachangzhang/eb_vjepa/scripts/run_action_train_smoke.sh
SMOKE_GPUS=5,6 SMOKE_WORLD=2 SMOKE_TAG=train_smoke_final_ddp bash /data/users/jiachangzhang/eb_vjepa/scripts/run_action_train_smoke.sh
SMOKE_GPUS=7 SMOKE_BATCH=32 SMOKE_TAG=train_smoke_final_batch32 bash /data/users/jiachangzhang/eb_vjepa/scripts/run_action_train_smoke.sh
bash /data/users/jiachangzhang/eb_vjepa/scripts/run_learned_smoke.sh --output /data/users/jiachangzhang/eb_vjepa/experiments/action_only/clean_learned_smoke.json
LIBERO_SOURCE=/data/users/jiachangzhang/eb_vjepa/repos/LIBERO-PRO-official LIBERO_CONFIG_PATH=/data/users/jiachangzhang/eb_vjepa/experiments/action_only/pro_config bash /data/users/jiachangzhang/eb_vjepa/scripts/run_learned_smoke.sh --suite libero_spatial_lan --output /data/users/jiachangzhang/eb_vjepa/experiments/action_only/pro_learned_smoke.json
LIBERO_SOURCE=/data/users/jiachangzhang/eb_vjepa/repos/LIBERO-plus-official LIBERO_CONFIG_PATH=/data/users/jiachangzhang/eb_vjepa/experiments/action_only/plus_config bash /data/users/jiachangzhang/eb_vjepa/scripts/run_learned_smoke.sh --output /data/users/jiachangzhang/eb_vjepa/experiments/action_only/plus_learned_smoke.json
```

## Production cache blocked by invalid source video

Full-cache supervisor PID 829354 used GPUs 1-4 and stopped all its owned workers
after goal episode 82 failed. No data was skipped, substituted or modified.
Completed episode counts before stopping: spatial 81, object 68, goal 82, 10 35.
No full-suite metadata or production normalization acceptance is claimed.
Status: full_cache_status.json phase=failed. The first earlier launcher attempt
failed before workers due to full system /tmp; it was archived and TMPDIR fixed.

Invalid source:
`/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.wrist_image/episode_000082.mp4`

Container reports 129 frames, PyAV/libdav1d decodes 105 then raises
InvalidDataError. Independent ffmpeg 7 libaom-av1 also reports
"Corrupt frame detected" / "Failed to decode tile data". Agent camera decodes
all 129 frames. Wrist file size is 243411 bytes, SHA256:
b3788859f4e52d882478ef3b2ffdc73d8419508e245940430efc047bd41ef732.
This matches the existing Hugging Face download metadata, revision
222cf888ed360fad0a5f983748c1cc40743d43e7. Re-downloading the same bytes would
not repair the stream.

Need an explicitly approved provenance-preserving data repair or exclusion
policy; do not silently drop this episode or conceal corruption with frames.
Formal training remains unlaunched, ready_plan.json absent. Full 8-GPU
production-batch acceptance and all-task benchmark evaluations remain pending.

Read-only full video audit completed: all 3386 camera videos decoded, 75.21 s,
exactly one failure (the goal wrist episode 82 above), 3385 successful.
Report: video_audit.json; command:

```bash
TMPDIR=/data/users/jiachangzhang/eb_vjepa/cache/tmp /data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/python /data/users/jiachangzhang/eb_vjepa/scripts/audit_libero_videos.py
```

No GPU jobs remain from preparation after the failed cache's owned workers
exited. Requested user decision: allow explicit exclusion of the entire bad
episode (both cameras/actions/states aligned), or retain full data and repair
it from a verified source. No exclusion has been implemented or authorized yet.

## Authorized exclusion and cache restart: 2026-10-03

The user explicitly authorized excluding the complete LIBERO-Goal local episode
82. The original public dataset remains untouched. Manifest:
`experiments/libero_episode_selection.json`.

The manifest excludes both camera streams and the aligned actions/states for
that one episode. Real parquet validation passed: 1692 selected episodes,
273336 frames, 129 excluded frames, and original_data_modified=false.
Selection SHA256:
`224606072f6dfa1d3a258fe2080cd46978da890fc8e435bcc7bc0b9b28e7e416`.
Report: `experiments/action_only/episode_selection_validation.json`.

Cache metadata, latent normalization statistics, model metadata, and resume
contracts now require the same selection SHA256. A new cache directory is
used: `cache_final_030000_exclude_goal82`. The failed pre-exclusion state is
retained as `full_cache_status_failed_corrupt_video.json`; a new four-GPU
cache job was launched with PID 2249988. Formal downstream training remains
unlaunched until this cache and normalization finish.

## Production 8-GPU launch: 2026-10-03 12:17 +08:00

Supersedes the earlier blocked/unlaunched entries above. The selected final-EMA
cache and normalization completed at 11:35:08: 1692 episodes, 273336 frames.
All eight GPUs were idle before the user explicitly authorized this launch.

Single production run:
- Supervisor PID: 3091871; torchrun PID: 3091874.
- Launcher: `scripts/launch_action_final.py`; exclusive `formal_train.lock`.
- Status: `experiments/action_only/formal_train_status.json`.
- Log: `experiments/action_only/train_final.log`.
- Output/checkpoints: `experiments/action_only/train_final`.
- W&B project `eb-vjepa`, run `libero_action_only_ctp_final_030000`, ID `b785369c`, online.
- Frozen CTP step-30000 EMA encoder; real LIBERO language embeddings; action-only.
- Eight DDP ranks, batch 32/rank, accumulation 1, global batch 256.
- 20 epochs, 21360 optimizer steps; upstream policy peak LR 2e-4 with warmup/cosine.
- Save every 5000 steps and at completion, with resumable training state.
- Goal episode 82 explicitly excluded using the manifest and hashes above.

Exact supervisor launch:

```bash
/data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python /data/users/jiachangzhang/eb_vjepa/scripts/launch_action_final.py --detach
```

Exact child command, run from `/data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer`:

```bash
/data/users/jiachangzhang/eb_vjepa/envs/action_only_test/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 /data/users/jiachangzhang/eb_vjepa/repos/RoboActualizer/scripts/train.py task=libero_action_only_final batch_size=32 num_workers=4 num_epochs=20 gradient_accumulation_steps=1 wandb.name=libero_action_only_ctp_final_030000 hydra.run.dir=/data/users/jiachangzhang/eb_vjepa/experiments/action_only/train_final/hydra
```

The launcher's source records the exact environment, including GPU IDs 0-7,
OMP_NUM_THREADS=4, data-disk temporary/cache paths and PYTHONPATH.
Startup acceptance at 12:19:10: step 180/21360, finite loss 0.6876,
image_loss=0, 2.08 optimizer steps/s (531 samples/s), estimated remaining
2h50m. All eight GPUs have worker allocations around 10189 MiB.
This early ETA is not a completion guarantee. W&B confirmed online syncing.

The existing watcher has no ready_plan.json and is not the owner of this run.
Do not create a competing train plan or launch another training job.
Full LIBERO/Plus/Pro benchmark evaluation is still pending; the formal
supervisor currently manages training only, not automatic evaluation.

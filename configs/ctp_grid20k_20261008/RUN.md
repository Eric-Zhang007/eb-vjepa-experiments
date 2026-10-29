# DROID 1k-20k Checkpoint Grid

## Live Audit: 2026-10-09 08:58 Asia/Shanghai

Search PID3674611 remains active. No duplicate work was launched. User-stopped
pretraining remains stopped at17358. All17full1k-17k checkpoints have completed
verified E: archive receipts; all corresponding EMA exports remain on server.

Completed matched results (actual report counts, unique episode keys and action
checkpoint SHA256 revalidated with `completed_results`):

| Encoder | Action checkpoint | Clean LIBERO | LIBERO-Plus |
| --- | --- | --- | --- |
| 1k | Fixed20k | 127/200 (63.50%) | 127/280 (45.36%) |
| 9k | Fixed20k | 147/200 (73.50%) | 144/280 (51.43%) |
| 1k | Lowest train loss,21360 | 137/200 (68.50%) | 126/280 (45.00%) |
| 9k | Lowest train loss,21360 | 143/200 (71.50%) | 145/280 (51.79%) |

Primary equal-weight scores:1k54.4286%,9k62.4643%.9k is the best observed
candidate so far, not an established peak. These are exploratory subsets.
The17k encoder cache and normalization completed; its action training was at
1570/21360, loss0.06995, approximately2.01steps/s. Formal supervisor2707036,
torchrun2707041, four ranks, physical GPUs0-3, batch64/rank, global256.
Observed GPUs0-3 at99-100% utilization and17147MiB each; GPUs4-7 idle at5MiB.
No intervention needed. Continue the existing search, not historical eight-GPU
or Pro queues.17k training ETA is roughly2.7hours plus subsequent evaluations.
Search is unfinished, so the monitoring automation remains enabled.

## Superseding Update: 2026-10-08 20:50 Asia/Shanghai

The user explicitly stopped pretraining and restricted the search to1k-17k.
Rank0 received SIGTERM; the trainer propagated the stop, saved final at17358,
reported `sanity_passed_all_ranks=true`, and all8GPU workers exited normally.
The off-grid17358 final is preserved but is not a search candidate.

The search now owns ONLY physical GPUs0,1,2,3. Action training uses4DDP ranks,
batch64 per GPU, accumulation1, unchanged global batch256. Cache generation
uses16workers total (4/GPU); evaluation preserves8protocol shards mapped two
per owned GPU. GPUs4-7 are not used or required to be idle by this pipeline.
All other model/data/optimizer settings remain unchanged; changing rank count
and local batch preserves the recipe but does not imply bitwise-identical RNG.

Search grid:1k..17k; first midpoint9k, endpoint anchors1k/17k, then guarded
binary interval narrowing and1k-neighbor checks. Primary score remains the
equal-weight clean200/Plus280 mean with the fixed20k action checkpoint.
Lowest-training-loss action selection remains secondary.

Current supervisor PID3674611, command:

```bash
/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/python /data/users/jiachangzhang/eb_vjepa/scripts/run_ctp_grid_search.py --detach --resume --max-encoder-step 17000 --allow-saved-early-stop --gpus 0,1,2,3 --batch-size 64
```

Checkpoint availability permits either a verified local archive receipt or a
live full checkpoint. Live checkpoints are tensor-compared to the EMA export
by cache preparation; archived ones use hash-bound source audits. This avoids
blocking9k on the remaining17k archive transfer. Archive worker PID20908 now
uses `archive_ctp_grid_local.py --max-step 17000`; a stalled17k SCP process was
stopped without removing its server source and transfer is being retried.

The first9k preparation attempt failed before GPU work because the action
environment lacked the inherited PyTorch PYTHONPATH. The search now explicitly
sets PYTHONPATH/TMPDIR/CUDA_VISIBLE_DEVICES for its child. Failed logs were
preserved in `failed_environment_009000_1791464136`; no checkpoint or cache
payload was present or deleted. Do not treat this attempt as a training result.

Verification:11peak/archive-contract tests,6launcher tests including4x64 with
GPUs4-7 occupied, and2mocked continuation tests including4GPU shard mapping
passed. Actual batch64 training awaits the9k cache; no completed64/GPU smoke
or downstream success rate is claimed yet.

Live acceptance after the environment fix: search state `running_candidate`,
candidate9k state `preparing_cache`, cache state `encoding`;16workers mapped
only to GPUs0-3 have written67real episode payloads. GPUs0-3 were100% utilized
at about9.3GiB each; GPUs4-7 were idle at5MiB each. This verifies cache launch,
not yet batch64 action-training completion.

All historical20k/eight-GPU statements below are superseded by this section.

## Ownership and Authorization

Requested on 2026-10-08: stop the remaining downstream queue, repeat the full-DROID continued-pretraining recipe from the same official checkpoint, save every 1k through 20k, and stop at 20k. Keep the original 72h wall-clock cosine schedule. Then search downstream performance at 1k resolution.

The former 50k downstream trial and queue were cancelled, not completed. Their model checkpoints, logs and prior evaluation results are preserved. Its explicitly authorized latent payload retirement removed 219,441,810,840 bytes; all cache identity/statistics JSON files remain. No old pretraining checkpoint was deleted.

The user separately authorized archiving only this new run's complete training-state snapshots to `E:/eb_vjepa_archive/`. A server snapshot may be unlinked only after its local copy passes a full SHA256 check. All new EMA exports and the 20k complete state remain on the server. Old models remain untouched.

## Pretraining

- Server root: `/data/users/jiachangzhang/eb_vjepa`.
- Output: `experiments/ctp_grid20k_20261008` under that root.
- Started: 2026-10-08 01:42:11 UTC / 09:42:11 Asia/Shanghai.
- Torchrun PID: `2551809`; eight DDP workers, no concurrent GPU job.
- W&B project: `eb-vjepa`; online run ID: `6gl8liap`.
- W&B name: `droid_vitl16_256_16f_ctp_grid20k_20261008`.
- Source: official `checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt`, both encoders initialized using `ema_encoder`. Not resumed from a previously trained encoder.
- Same 12-block predictor trunk and explicit native-output-head adaptation as the preceding full-DROID run.
- Microbatch: 16/GPU; accumulation: 1; global batch: 128; activation checkpointing off.
- Same three-view DROID manifest, outcome-proportional sampling, PyAV decode, 256px/16 frames/4fps video recipe, seed239, AdamW, WD0.04 and EMA0.99925.
- LR: step warmup for500 steps from2e-6 to2e-5, then wall-clock cosine to2e-6 at72h. `max_steps=1000000`, `wall_seconds=259200` and `lr_clock=wall` remain unchanged.
- Existing context-loss schedule is unchanged: weight0 through15k, linear increase toward0.5 at30k. This run stops during that ramp, at20k.
- Only config differences: run name, output location, `snapshot_every: 1000`. The independent CLI stop is `--stop-step 20000`.
- `latest.pt` every500 steps; complete `step_001000.pt` through `step_020000.pt`; corresponding EMA exports; `final.pt` at20k. Atomic saves and hard links preserve resumability. Early low-disk/signal/72h termination is treated as an incomplete grid, never as permission to launch downstream.
- Training implementation and manifest hashes match the prior real100-step resume smoke. Existing actual calibration/smoke evidence was reused with provenance in `validation_reuse.json`; no new smoke run is claimed. Ongoing per-step gradient/EMA/finite checks remain enabled.
- Initial observed100-step window: about2.05seconds/optimizer step. Allow roughly12-14hours total with checkpoint I/O and changing step costs; this is an estimate, not a deadline. More frequent saves affect elapsed wall time but do not change the LR formula.

Exact launch command (environment recorded in `launch.json`):

```bash
/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/python /data/users/jiachangzhang/eb_vjepa/ctp_full/launch.py --config /data/users/jiachangzhang/eb_vjepa/experiments/ctp_grid20k_20261008/config.yaml --stop-step 20000 --launch
```

The generated formal command uses `torchrun --standalone --nproc_per_node=8 .../ctp_full/train.py --config .../calibrated.yaml --stage train --stop-step 20000`. Log: `experiments/ctp_grid20k_20261008/train.log`; per-step metrics: `metrics.jsonl`.

## Verified Archive

- Windows archive worker PID: `39200`; script: `scripts/archive_ctp_grid_local.py`.
- Local directory: `E:/eb_vjepa_archive/ctp_grid20k_20261008/`.
- Local state/logs: `status.json`, `archive_stdout.log`, `archive_stderr.log`.
- Server verifies each EMA export equals its complete checkpoint,302 tensors are finite, checkpoint step/sanity match, and binds both files by SHA256.
- SCP writes a partial local file; size/hash verification and fsync precede final rename and server commit. Failed or incomplete transfers do not authorize server deletion.
- Server records `checkpoint_audit_<step>.json` and `archive_receipt_<step>.json`. Every full snapshot remains recoverable locally;20k is additionally retained remotely.
- The archive process requires this Windows machine and E: drive to remain online. An archive interruption blocks downstream until all20 receipts are complete; it does not stop or delete the active pretraining run.

## Automatic Downstream Search

- Supervisor PID: `467846`; lock/state: `experiments/action_only/ctp_grid20k_search/search.lock` and `status.json`.
- At launch it is waiting for pretraining, not consuming GPUs.
- It requires normal20k completion, sanity pass, verified archives/EMA identities for all20 checkpoints, and idle GPUs before launching a candidate.
- Same downstream recipe as prior matched experiments: frozen EMA encoder, action-only head, real cached language, same action normalization/data exclusion,20epochs,8GPUs, global batch256. No action predictor or language substitutions.
- Each encoder gets fresh encoder-specific latents and channel statistics. Cache encoding uses4workers/GPU; normalization uses32CPUworkers. All model loads and cache identities remain checked.
- Storage only: spatial/object/goal latent payloads use a private `/dev/shm/eb_vjepa_grid_<uid>/encoder_<step>` directory; LIBERO10 remains on the data disk. Approximately138GB RAM payload is budgeted with a10% margin, at least40GB spare tmpfs and100GB MemAvailable headroom. Tensor format and training inputs are unchanged. Cache is reconstructible after reboot. Only a completed trial's newly generated payloads are retired, with metadata copied to persistent storage.
- Candidate directory: `experiments/action_only/ctp_grid20k_search/encoder_<step>`.
- W&B training name: `libero_action_only_grid20261008_<step>` in project `eb-vjepa`. Closed-loop results are verified local JSON/Markdown files, not claimed to be W&B-uploaded.
- Primary comparison: fixed20k action-head checkpoint. Lowest-training-loss saved action checkpoint is evaluated separately and never substituted into the primary search score.
- Primary score: equal-weight mean of clean LIBERO200-episode success rate and LIBERO-Plus280-episode success rate. Same pre-existing exploratory subsets and text-CFG1.0.
- Anchors:10k,1k,20k. Binary adjacent-midpoint comparisons shrink the interval when the observations support that direction; the winning point's available +/-1k neighbors are checked.
- Non-unimodal observations or adjacent scores within1percentage point disable the unsafe interval discard and trigger denser guarded evaluation, up to the20-point grid.1pp is an operational ambiguity threshold, not a confidence interval. The observed best can be noisy; a binary result is conditional on unimodality and is not a proven global/statistical optimum.
- This is benchmark-subset-driven exploratory model selection. Selected performance is not an untouched held-out generalization estimate.
- Every candidate runs serially on8GPUs; no competing trainers. PID/lock/state checks refuse duplicate or incomplete trial restarts. Failures are recorded and do not silently skip candidates.
- Reports: `results.json`, `scores.json`, `decision.json`, `RESULTS.md`, `search.log` under the search directory.

Exact supervisor command:

```bash
/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/python /data/users/jiachangzhang/eb_vjepa/scripts/run_ctp_grid_search.py --detach --resume
```

The initial supervisor was restarted while still waiting to apply final safety checks; `--resume` did not resume/restart any model training. A dormant old `watch_action_pipeline.py` owner was also stopped to prevent stale automatic launches.

## Verification

- Ten CPU unit tests passed for all20 single-peak locations, endpoint peaks, plateaus, non-unimodality handling, EMA hash/receipt rejection, and exact prior-gate reuse constraints.
- Six tiny-tensor CPU archive tests passed: torch serialization, EMA/source audit, wrong-hash and wrong-destination rejection, idempotent archive commit,20k retention and early-stop detection. These fixtures are not real evaluation results.
- Five existing launch-contract tests and one full mocked continuation test passed after adding the grid source/name options.
- Three hybrid-storage tests passed for symlink-aware discovery, preserving models/metadata during owned-payload retirement, memory headroom rejection and refusing unrelated symlink targets. Total:25 unit/contract tests.
- Existing8-rank proportional sampling/replay, wall-clock cosine/resume and actual DROID-manifest checks also passed. Official source SHA256 and upstream commit match the prior run, recorded in `source_identity_verification.json`.
- Real8-GPU pretraining has produced finite losses, nonzero encoder gradients, nonzero EMA updates and online W&B logs. The first actual1k archive and new-encoder downstream results are pending; no completed transfer or downstream score is claimed at launch.

Do not rerun the preparation command or create a second training/search owner. Read the persistent states first. Do not restart the old encoder candidate queue.

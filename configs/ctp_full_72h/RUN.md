# Full DROID 72h Formal Run

Started (UTC): 2026-10-03T10:02:13.744196+00:00

Started (Asia/Shanghai): 2026-10-03 18:02:13.
Expected wall-cap completion: approximately 2026-10-06 18:02 +08:00,
plus final checkpoint writing. This is a 72-hour run, not another 30k-step run.

PID / session leader: 3204721

Log: `/data/users/jiachangzhang/eb_vjepa/experiments/ctp_full_72h/train.log`

W&B: `{'id': 'rg58nlvr', 'url': 'https://wandb.ai/zjc010100000-xi-an-jiaotong-university-/eb-vjepa/runs/rg58nlvr', 'mode': 'online'}`

Microbatch/GPU: 16; accumulation: 1; effective global batch: 128.

Activation checkpointing: False.

Measured clips/sec: 50.716; seconds/optimizer-step: 2.524; projected 72h steps: 102699.

Stop at 72 hours (1000000-step safety ceiling). Save latest, every 5k, 24h/48h/72h and final. Low disk (<12 GiB) triggers a final safety save.

Full frozen three-camera manifest with is_episode_successful labels used only for proportional global-batch sampling, never model supervision. Original episode-class proportions are preserved with cumulative rounding. The fixed clip requires 3.75 seconds plus one original-frame margin; shorter episodes are excluded, as audited in data_report.json.

AdamW peak LR 2e-5, 500-step warmup, then wall-clock cosine to 2e-6 at 72h; WD 0.04; EMA 0.99925; seed 239. Not resumed from CTP-v1.

Both encoders initialized from ema_encoder. Predictor trunk reused; two teacher-space output heads explicitly reinitialized for native 1024-dim EMA targets. See load_report_rank0.json for exact reuse percentages.

Exact command (environment saved in launch.json):

```bash
/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/torchrun --standalone --nproc_per_node=8 /data/users/jiachangzhang/eb_vjepa/ctp_full/train.py --config /data/users/jiachangzhang/eb_vjepa/experiments/ctp_full_72h/calibrated.yaml --stage train
```

## Data and Sampling Audit

- Complete metadata: 95658 contiguous episode IDs; success 78864, failure 16794.
- Target ratio by original episodes: success 82.4437057%, failure 17.5562943%.
- Each global batch: 105/23 or 106/22 success/failure clips.
  Cumulative rounding error is at most half a sample.
- Download inventory: 813 videos; complete episode metadata references 784.
  All referenced videos exist. The 29 unreferenced downloaded files are listed
  in data_report.json and are not assigned guessed boundaries/outcomes.
- Eligible data: 93919 episodes, all three views, 281757 episode-camera samples.
  Fixed-length clip filtering excludes 130 successful and 1609 failed short
  episodes. Sampling still preserves the ORIGINAL ratio, not the filtered ratio.
- Manifest SHA256:
  `0b5ce2d2035d0aa35fd3a97b23aa43869a84dcc5f5167a9ff3425e57998ff630`.
- No action, language, proprioception or reward values are loaded for training.
  Success/failure labels are used only by the sampler, never as model inputs.

## Startup Acceptance

CPU tests passed for 8-rank sample partitioning, accumulation, proportional
rounding, exact sample/augmentation-seed replay, and wall-clock LR resume/endpoints.
One bounded activation-checkpoint comparison: 40 steps ON and 40 OFF.
OFF selected: 50.716 clips/s versus 46.397 ON.

One 100-step smoke, split 90+10 with actual checkpoint/resume:
`smoke/pass.json` passed=true, sanity_passed_all_ranks=true, resumed_from=90,
step=100; first resumed loss 0.631733, last 0.627593.
The smoke passed using the final trainer hash; the only post-calibration trainer
change stores wall-hour EMA exports without retaining additional full optimizer
snapshots. Periodic 5k full snapshots remain unchanged.

Formal step 25 verified: loss 0.664260, finite nonzero encoder gradient,
EMA delta 1.8626e-8, 105 success and 23 failure clips. Throughput at that step
50.434 clips/s, 2.538 s/step, data-and-mask wait 0.047 s.
Live nvidia-smi snapshot: all GPUs 0-7 at 100% utilization.
W&B mode online, project eb-vjepa, ID rg58nlvr.

Both encoders load 100% from the official ema_encoder. The released predictor
loads strictly before native-head adaptation; its 12-block trunk is reused
100%, retaining 94.4227% of original predictor parameters overall.
Only the two teacher-space output heads are explicitly reinitialized.

The run does not resume CTP-v1, the smoke, or any downstream checkpoint.
Previous CTP-v1 and LIBERO artifacts are unchanged. Downstream ownership
documents now prohibit competing GPU jobs during this experiment.

## Checkpoint and Ownership Notes

Latest resumable state every 500 steps; permanent full state and EMA export
every 5000 steps; EMA-only aliases at 24h/48h/72h plus final full checkpoint.
Wall-hour JSON markers record exact step and elapsed seconds.
Every full checkpoint contains online/EMA/predictor, optimizer, scheduler,
all eight RNG states, sampler position via step, and immutable resume contract.

Supervisor acceptance PID 131415, launch-helper PID 131416; formal torchrun
PID 3204721. pipeline_status.json records the transition to training.
run.lock protects the formal output; duplicate launch guards and GPU-idle gates
prevent another launch through this pipeline. Check process and summary.json
as well as pipeline_status.json when determining whether training has finished.

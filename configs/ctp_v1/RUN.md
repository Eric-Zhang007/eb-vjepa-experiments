# CTP-v1 Formal Run

Started (UTC): 2026-10-02T03:58:51.370164+00:00

Started (Beijing): 2026-10-02 11:58:51 +08:00.

PID / session leader: 2421560

Log: `/data/users/jiachangzhang/eb_vjepa/experiments/ctp_v1/train.log`

W&B: `{'id': 'g948fxhq', 'url': 'https://wandb.ai/zjc010100000-xi-an-jiaotong-university-/eb-vjepa/runs/g948fxhq', 'mode': 'online'}`

Microbatch/GPU: 16; accumulation: 1; effective global batch: 128.

Activation checkpointing: False.

Measured clips/sec: 58.132; seconds/optimizer-step: 2.202; projected 72h steps: 30000.

Stop at 30,000 optimizer steps or 72 hours. Save latest, every 5k, and final.

At the calibration rate, 30k steps take approximately 18.35 hours before
checkpoint/logging overhead, so the step cap is expected to occur before
72 hours. This is a throughput projection, not a guaranteed completion time.

Verification: one bounded calibration (six sequential candidates, 40 steps each);
one 100-step smoke split at step 90 for a real checkpoint/resume. All eight
ranks passed finite-loss, online-gradient, no-target-gradient and EMA checks.
Smoke loss was 0.66830 at step 1 and 0.62607 at step 100.
Reports: `calibration/selection.json`, `smoke/pass.json`.

Dataset snapshot: 154 completed MP4s, 48,055 eligible episode-camera samples,
currently ONLY `observation.images.exterior_1_left`. This is not the full
three-camera DROID dataset. The frozen manifest is not refreshed during
training. Decoding is PyAV RGB24, with 16 frames on a 4Hz grid strictly within
episode boundaries. No action, language, proprioception, reward or task labels
are loaded.

First live verification: formal optimizer step 17, loss 0.661865, all eight
GPUs at 100% utilization in the nvidia-smi snapshot. No competing training run.

Both encoders initialized from ema_encoder. Predictor trunk reused; two teacher-space output heads explicitly reinitialized for native 1024-dim EMA targets. See load_report_rank0.json for exact reuse percentages.

Exact command (environment saved in launch.json):

```bash
/data/users/jiachangzhang/eb_vjepa/envs/eb_vjepa_pretrain/bin/torchrun --standalone --nproc_per_node=8 /data/users/jiachangzhang/eb_vjepa/ctp_v1/train.py --config /data/users/jiachangzhang/eb_vjepa/experiments/ctp_v1/calibrated.yaml --stage train
```

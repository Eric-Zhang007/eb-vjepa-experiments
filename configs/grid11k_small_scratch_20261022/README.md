# Small grid11k scratch experiment

Independent experiment; no prior action-head or world-model checkpoint is loaded.

## Recipe

- Frozen grid11k visual encoder, unchanged small predictor/action architecture (approximately 63.6M trainable parameters; the new two-view embedding adds 768 parameters).
- Two 224x224 views are encoded independently. Horizontal concatenation is only a transport format; it is split before the encoder. View-major tokens and learned camera embeddings are used downstream.
- Frozen Wan umT5-XXL, context length 128; existing exact-prompt embeddings copied into text_cache_umt5 (105 prompts). User authorized this fallback because Google T5 download was slow.
- Random predictor/action initialization, seed 42.
- Action horizon 32, replan every 16 environment actions, four denoising steps for the scheduled evaluation.
- Four future latent targets at offsets +8, +16, +24, +32; per-token LayerNorm on the frozen future target, masked element-wise L1. Current latent inputs remain raw.
- Action flow MSE + future L1, weights 1:1.
- Global batch 128 = four GPUs x 32, no accumulation. Training GPUs 0-3; GPUs 0-5 used for parallel visual preparation. Six equal DDP batches cannot divide 128 exactly.
- 21,360 optimizer updates; AdamW peak LR 1e-4, 1,068-step warmup, cosine decay to 1e-6, weight decay 0.01, clip norm 1.0.
- Existing data selection, action/proprio normalization, four-frame history with interval 2, resize-only visual preprocessing are retained. This is not a full official V-JEPA Policy reproduction. Existing selected dataset contains 1,692 episodes, not a claimed reproduction of their 1,693-episode list.

## Files and commands

Resolved configuration: `train_config.yaml`. Remote root: `/data/users/jiachangzhang/eb_vjepa`.

The running supervisor is `scripts/run_scratch_dualview.py`. It waits for the independent visual and T5 caches, runs real-cache preflight, tests four-rank pause/resume, then launches the fresh formal run. Smoke weights are never imported into formal training.

```bash
ROOT=/data/users/jiachangzhang/eb_vjepa
export PYTHONPATH=$ROOT/repos/RoboActualizer/src:$ROOT/envs/eb_vjepa_pretrain/lib/python3.11/site-packages
CUDA_VISIBLE_DEVICES=0,1,2,3 $ROOT/envs/action_only_test/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 $ROOT/repos/RoboActualizer/scripts/train.py --config-path $ROOT/experiments/grid11k_small_scratch_20261022 --config-name train_config
```

Do not run that command concurrently with the supervisor. Supervisor status is `status.json`; visual preparation is `cache_status.json`; text download/cache receipts are `text_download_status.json` and `text_cache_status.json`. Download supports byte-range resumption.

W&B project: `eb_vjepa`; run name `grid11k-small-scratch-dualview-umt5-l1-h32`. A run ID is not available until formal training starts. Evaluation writes local results only.

## Verification at setup

Seven existing predictive-context tests and two new camera-isolation/L1 tests passed. Real-cache/full-batch and four-rank resume tests are queued behind input preparation; they are not yet claimed as passed. Their receipts will be `preflight/results.json` and `ddp_resume_test.json`.

Final checkpoint is scheduled for the existing clean200/Plus280 exploratory evaluation interface, with replan16 and four denoising steps. These subsets are not official full-benchmark scores.


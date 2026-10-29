#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export TMPDIR="$ROOT/cache/tmp"
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTHONPATH="$ROOT/repos/RoboActualizer/src:$ROOT/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"
OUT="$ROOT/experiments/action_only/cache_smoke_020000"
DATA=/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot
"$ROOT/envs/action_only_test/bin/python" "$ROOT/repos/RoboActualizer/scripts/precompute_rgb_latents.py" \
    task=libero_action_only encoder_tag=diagnostic_020000 \
    "model.backbone.ckpt_path=$ROOT/experiments/ctp_v1/embodied_vjepa_encoder_020000.pt" \
    "data.train.dataset_dirs=[$DATA]" \
    "data.train.rgb_latent_cache_dir=$OUT" model.latent_norm_stats_path=null \
    "+rgb_latent_cache_dir=$OUT" +encode_device=cpu +encode_cpu_threads=2 \
    +encode_batch_size=2 +max_episodes_per_suite=1 \
    "hydra.run.dir=$OUT/hydra"
"$ROOT/envs/action_only_test/bin/python" "$ROOT/repos/RoboActualizer/scripts/compute_latent_norm_stats.py" "$OUT"

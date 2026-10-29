#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec 9>"$ROOT/experiments/action_only/acceptance.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TMPDIR="$ROOT/cache/tmp"
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints/text"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export PYTHONPATH="$ROOT/repos/RoboActualizer/src:$ROOT/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"
PY="$ROOT/envs/action_only_test/bin/python"
REPO="$ROOT/repos/RoboActualizer"
OUT="$ROOT/experiments/action_only"
DATA=/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA
ENC="$ROOT/experiments/ctp_v1/embodied_vjepa_encoder_030000.pt"
CACHE="$OUT/cache_smoke_final"
BASE=(task=libero_action_only encoder_tag=ctp_final_030000
    "model.backbone.ckpt_path=$ENC"
    "data.train.dataset_dirs=[$DATA/libero_spatial_no_noops_1.0.0_lerobot]"
    "data.train.rgb_latent_cache_dir=$CACHE"
    "data.train.text_embedding_cache_dir=$OUT/text_cache")
echo "FINAL CACHE START $(date -Is)"
"$PY" "$REPO/scripts/precompute_rgb_latents.py" "${BASE[@]}" \
    model.latent_norm_stats_path=null "+rgb_latent_cache_dir=$CACHE" \
    +encode_batch_size=8 +max_episodes_per_suite=1 "hydra.run.dir=$CACHE/hydra"
"$PY" "$REPO/scripts/compute_latent_norm_stats.py" "$CACHE"
echo "TEXT CACHE START $(date -Is)"
"$PY" "$REPO/scripts/precompute_text_embeds.py" "${BASE[@]}" \
    "data.train.dataset_dirs=[$DATA/libero_spatial_no_noops_1.0.0_lerobot,$DATA/libero_object_no_noops_1.0.0_lerobot,$DATA/libero_goal_no_noops_1.0.0_lerobot,$DATA/libero_10_no_noops_1.0.0_lerobot]" \
    "model.latent_norm_stats_path=$CACHE/latent_norm_stats.json" "hydra.run.dir=$OUT/text_hydra"
echo "CACHE PREPARATION PASS $(date -Is)"

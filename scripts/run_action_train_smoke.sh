#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${SMOKE_TAG:-train_smoke_final}"
exec 9>"$ROOT/experiments/action_only/$TAG.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES="${SMOKE_GPUS:-0}"
export OMP_NUM_THREADS=4
export TMPDIR="$ROOT/cache/tmp"
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTHONPATH="$ROOT/repos/RoboActualizer/src:$ROOT/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"
PY="$ROOT/envs/action_only_test/bin/python"
OUT="$ROOT/experiments/action_only"
RUN="$OUT/$TAG"
WORLD="${SMOKE_WORLD:-1}"
LAUNCH=("$PY")
if [ "$WORLD" != 1 ]; then
    LAUNCH+=(-m torch.distributed.run --standalone "--nproc_per_node=$WORLD")
fi
ARGS=(task=libero_action_only encoder_tag=ctp_final_030000
 "model.backbone.ckpt_path=$ROOT/experiments/ctp_v1/embodied_vjepa_encoder_030000.pt"
 "data.train.dataset_dirs=[/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot]"
 "+data.train.episode_ids_file=$ROOT/experiments/action_only_episode_smoke.json"
 "data.train.rgb_latent_cache_dir=$OUT/cache_smoke_final"
 "data.train.text_embedding_cache_dir=$OUT/text_cache"
 "model.latent_norm_stats_path=$OUT/cache_smoke_final/latent_norm_stats.json"
 "output_dir=$RUN" "hydra.run.dir=$RUN/hydra"
 "batch_size=${SMOKE_BATCH:-4}" num_workers=0 max_steps=8 save_every=4 log_every=1
 "wandb.name=libero_action_only_${TAG}")
"${LAUNCH[@]}" "$ROOT/repos/RoboActualizer/scripts/train.py" "${ARGS[@]}"
# Resume the step-4 state with the same schedule and data contract.
"${LAUNCH[@]}" "$ROOT/repos/RoboActualizer/scripts/train.py" "${ARGS[@]}" "resume=$RUN/checkpoints/state/previous"
echo "REAL TRAIN AND RESUME PASS $(date -Is)"

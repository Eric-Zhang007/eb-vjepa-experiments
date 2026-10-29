#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/repos/Dream-VLX/vla"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1 WANDB_MODE=online
exec "$ROOT/envs/dream_vla_official/bin/python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=6 -m elaf.v0_train \
  --config "$ROOT/elaf/configs/v1.json" \
  --output "$ROOT/experiments/elaf_v1_grid11k_72token" "$@"

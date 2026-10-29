#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
cd "$ROOT/repos/Dream-VLX/vla"
export PYTHONPATH="$ROOT/repos/Dream-VLX/vla${PYTHONPATH:+:$PYTHONPATH}"
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export WANDB_PROJECT=eb_vjepa
export WANDB_MODE=online
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# One mixed-suite model, fixed global batch48 and200000 optimizer updates.
exec "$ROOT/envs/dream_vla_official/bin/python" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=6 vla-scripts/finetune_dream.py \
    --vla_path "$ROOT/checkpoints/dream_vla/Dream-VLA-7B" \
    --data_root_dir /data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/datasets/rlds \
    --dataset_name libero_mix4 \
    --run_root_dir "$ROOT/experiments/dream_vla_mix4" \
    --use_l1_regression False --use_diffusion False --use_flow_matching True \
    --use_film False --num_images_in_input 2 --use_proprio True \
    --batch_size 8 --learning_rate 1e-4 --num_steps_before_decay 100000 \
    --max_steps 200000 --save_freq 5000 --save_latest_checkpoint_only True \
    --image_aug True --lora_rank 32 --wandb_project eb_vjepa \
    --wandb_entity zjc010100000-xi-an-jiaotong-university- \
    --run_id_note Dream_VLA_mix4_200k_batch48 \
    "$@"

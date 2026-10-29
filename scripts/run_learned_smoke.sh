#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$ROOT/cache/imagemagick/root/usr/lib/x86_64-linux-gnu:$ROOT/cache/osmesa/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAGICK_CONFIGURE_PATH="$ROOT/cache/imagemagick/root/etc/ImageMagick-6"
export MAGICK_HOME="$ROOT/cache/imagemagick/home"
export TMPDIR="$ROOT/cache/tmp"
export MESA_SHADER_CACHE_DIR="$ROOT/cache/mesa"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$ROOT/cache/numba"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints/text"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/users/jiachangzhang/DNFT4RP/.libero-config}"
export PYTHONPATH="${LIBERO_SOURCE:-/data/users/jiachangzhang/DNFT4RP/upstream/libero}:$ROOT/repos/RoboActualizer/src:$ROOT/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"
exec "$ROOT/envs/action_only_test/bin/python" "$ROOT/repos/RoboActualizer/scripts/check_learned_closed_loop.py" \
    --run "$ROOT/experiments/action_only/train_smoke_final" "$@"

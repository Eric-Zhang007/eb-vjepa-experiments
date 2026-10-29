#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-7}" OMP_NUM_THREADS=2
export MUJOCO_EGL_DEVICE_ID="$CUDA_VISIBLE_DEVICES"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$ROOT/cache/imagemagick/root/usr/lib/x86_64-linux-gnu:$ROOT/cache/osmesa/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAGICK_CONFIGURE_PATH="$ROOT/cache/imagemagick/root/etc/ImageMagick-6"
export MAGICK_HOME="$ROOT/cache/imagemagick/home"
export MAGICK_CODER_MODULE_PATH="$ROOT/cache/imagemagick/root/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders"
export MAGICK_FILTER_MODULE_PATH="$ROOT/cache/imagemagick/root/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/filters"
export TMPDIR="$ROOT/cache/tmp" MESA_SHADER_CACHE_DIR="$ROOT/cache/mesa"
export MPLCONFIGDIR="$ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$ROOT/cache/numba"
export PYTHONDONTWRITEBYTECODE=1
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints/text"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/users/jiachangzhang/DNFT4RP/.libero-config}"
export PYTHONPATH="${LIBERO_SOURCE:-/data/users/jiachangzhang/DNFT4RP/upstream/libero}:$ROOT/repos/RoboActualizer/src:$ROOT/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"
exec "$ROOT/envs/action_only_test/bin/python" "$ROOT/scripts/${EVAL_SCRIPT:-eval_10k_libero_probe.py}" "$@"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$ROOT/cache/osmesa/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/users/jiachangzhang/DNFT4RP/.libero-config}"
export TMPDIR="$ROOT/cache/tmp"
export MESA_SHADER_CACHE_DIR="$ROOT/cache/mesa"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$ROOT/cache/numba"
export PYTHONPATH="${LIBERO_SOURCE:-/data/users/jiachangzhang/DNFT4RP/upstream/libero}:$ROOT/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"
exec "$ROOT/envs/action_only_test/bin/python" "$ROOT/scripts/inspect_downstream_assets.py" \
    --sim-smoke --output "$ROOT/experiments/action_only/unified_rgb_smoke.json" "$@"

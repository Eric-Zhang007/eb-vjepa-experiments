#!/bin/bash
# eb_vjepa environment verification (run on 112.45.47.40 as jiachangzhang)
# Usage: bash scripts/verify_env.sh [nccl_nproc]   (default 2)
set -u
ROOT=/data/users/jiachangzhang/eb_vjepa
PY=$ROOT/envs/eb_vjepa_pretrain/bin/python
NPROC=${1:-2}
cd "$ROOT/repos/vjepa2" || exit 1

echo "=== [1] python ==="
$PY -V
echo "=== [2] torch + CUDA ==="
$PY -c "import torch; print('torch', torch.__version__, '| cuda_build', torch.version.cuda, '| available', torch.cuda.is_available(), '| ndev', torch.cuda.device_count())"
echo "=== [3] GPUs visible ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=== [4] bf16 matmul on cuda:0 ==="
$PY -c "import torch; a=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16); b=torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16); c=a@b; print('bf16_ok', c.dtype, float(c.float().abs().mean()))"
echo "=== [5] repo imports ==="
$PY -c "import sys; sys.path.insert(0, '$ROOT/repos/vjepa2'); import src.datasets.video_dataset, src.models.vision_transformer, decord, pandas, timm, transformers; v=getattr(decord,'__version__','?'); print('imports OK | decord', v, '| timm', timm.__version__)"
echo "=== [6] NCCL $NPROC-proc allreduce ==="
"$ROOT/envs/eb_vjepa_pretrain/bin/torchrun" --standalone --nproc_per_node=$NPROC "$ROOT/scripts/verify_nccl.py"
echo "=== done ==="

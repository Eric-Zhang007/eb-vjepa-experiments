#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
ENV=$ROOT/envs/dream_vla_official
REPO=$ROOT/repos/Dream-VLX/vla
OUT=$ROOT/experiments/dream_vla_official/setup
mkdir -p "$OUT" "$ROOT/cache/tmp"
exec 9>"$OUT/install.lock"
flock -n 9 || exit 0
trap 'echo "$?" > "$OUT/exitcode"' EXIT
export TMPDIR="$ROOT/cache/tmp" HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export PATH="$ENV/bin:/usr/local/cuda-12.8/bin:$PATH" CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8
UV=/home/jiachangzhang/.local/bin/uv
# Wait only for our in-flight official torch transaction.
while pgrep -fu jiachangzhang 'uv pip install --python /data/users/jiachangzhang/eb_vjepa/envs/dream_vla_official/bin/python torch==2.5.1' >/dev/null; do sleep 15; done
echo torch > "$OUT/stage"
"$UV" pip install --python "$ENV/bin/python" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
echo python_dependencies > "$OUT/stage"
"$UV" pip install --python "$ENV/bin/python" --index-url https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.26.4 tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics==2021.12.3 tensorflow-metadata==1.15.0 transformers==4.51.3 peft==0.17.0 diffusers==0.35.1 accelerate==1.10.1 timm==0.9.10 draccus==0.8.0 sentencepiece==0.1.99 wandb==0.21.3 einops json-numpy jsonlines matplotlib rich imageio uvicorn fastapi ninja packaging psutil opencv-python==4.10.0.84
# Repo's editable package uses a Git dlimp dependency; install separately with configured proxy.
echo dlimp > "$OUT/stage"
if [[ ! -d "$ROOT/repos/dlimp_openvla/.git" ]]; then
 git -c http.proxy=http://127.0.0.1:7890 clone https://github.com/moojink/dlimp_openvla "$ROOT/repos/dlimp_openvla"
fi
"$UV" pip install --python "$ENV/bin/python" --index-url https://pypi.tuna.tsinghua.edu.cn/simple "$ROOT/repos/dlimp_openvla"
"$UV" pip install --python "$ENV/bin/python" --no-deps -e "$REPO"
echo flash_attention > "$OUT/stage"
python -m pip install flash-attn==2.5.5 --no-build-isolation --index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo verify_help > "$OUT/stage"
cd "$REPO"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=''
python vla-scripts/finetune_dream.py --help > "$OUT/official_help.txt" 2>&1
python - <<'PY'
import json,torch,transformers,tensorflow,peft,diffusers,flash_attn
from pathlib import Path
out=Path('/data/users/jiachangzhang/eb_vjepa/experiments/dream_vla_official/setup')
(out/'receipt.json').write_text(json.dumps({'status':'official_help_passed','torch':torch.__version__,'transformers':transformers.__version__,'tensorflow':tensorflow.__version__,'peft':peft.__version__,'diffusers':diffusers.__version__,'flash_attn':flash_attn.__version__,'training_started':False},indent=2))
PY
echo ready_for_official_smoke > "$OUT/stage"

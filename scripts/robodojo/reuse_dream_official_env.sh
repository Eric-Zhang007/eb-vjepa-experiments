#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
SOURCE=/data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/.venv
TARGET=$ROOT/envs/dream_vla_official
OUT=$ROOT/experiments/dream_vla_official/setup_reuse
mkdir -p "$OUT" "$ROOT/cache/tmp"
exec 9>"$OUT/install.lock"; flock -n 9 || exit 0
trap 'echo "$?" > "$OUT/exitcode"' EXIT
export TMPDIR="$ROOT/cache/tmp" HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export PATH="$TARGET/bin:/usr/local/cuda-12.8/bin:$PATH" CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8
UV=/home/jiachangzhang/.local/bin/uv
if [[ ! -f "$OUT/clone_receipt.json" ]]; then
 echo clone_existing_environment > "$OUT/stage"
 python3 - <<'PY'
from pathlib import Path
import os,subprocess,json,datetime,shutil
src=Path('/data/users/jiachangzhang/DNFT4RP/upstream/vla_adapter/.venv').resolve()
dst=Path('/data/users/jiachangzhang/eb_vjepa/envs/dream_vla_official')
assert dst.parent.resolve()==Path('/data/users/jiachangzhang/eb_vjepa/envs').resolve()
assert shutil.disk_usage(dst.parent).free>20*1024**3
if dst.exists():dst.rename(dst.with_name(dst.name+'.bootstrap_saved_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
subprocess.run(['cp','-a','--reflink=auto',str(src),str(dst)],check=True)
patched=[]
for path in list((dst/'bin').iterdir())+list((dst/'lib').rglob('*.pth'))+list((dst/'lib').rglob('*.egg-link'))+[dst/'pyvenv.cfg']:
 if not path.is_file() or path.is_symlink():continue
 try:text=path.read_text()
 except (UnicodeError,OSError):continue
 if str(src) in text:
  path.write_text(text.replace(str(src),str(dst)));patched.append(str(path))
a=src/'lib/python3.10/site-packages/torch/__init__.py';b=dst/'lib/python3.10/site-packages/torch/__init__.py'
assert a.stat().st_ino!=b.stat().st_ino,'Hardlinks forbidden'
out=Path('/data/users/jiachangzhang/eb_vjepa/experiments/dream_vla_official/setup_reuse')
(out/'clone_receipt.json').write_text(json.dumps({'source':str(src),'target':str(dst),'copy':'cp -a --reflink=auto','shared_inodes':False,'patched_paths':patched},indent=2))
PY
fi
echo torch_update > "$OUT/stage"
"$UV" pip install --python "$TARGET/bin/python" torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
echo required_updates > "$OUT/stage"
"$UV" pip install --python "$TARGET/bin/python" transformers==4.51.3 peft==0.17.0 diffusers==0.35.1 numpy==1.26.4 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo flash_attention > "$OUT/stage"
"$UV" pip install --python "$TARGET/bin/python" ninja packaging --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install flash-attn==2.5.5 --no-build-isolation --index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo official_editable_registration > "$OUT/stage"
"$UV" pip install --python "$TARGET/bin/python" --no-deps --reinstall -e "$ROOT/repos/Dream-VLX/vla"
echo official_help > "$OUT/stage"
cd "$ROOT/repos/Dream-VLX/vla"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
python vla-scripts/finetune_dream.py --help > "$OUT/official_help.txt" 2>&1
python - <<'PY'
import json,sys,torch,tensorflow,transformers,peft,diffusers,flash_attn,dlimp
from pathlib import Path
p=Path('/data/users/jiachangzhang/eb_vjepa/experiments/dream_vla_official/setup_reuse/receipt.json')
p.write_text(json.dumps({'status':'official_help_passed','python':sys.executable,'torch':torch.__version__,'tensorflow':tensorflow.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'diffusers':diffusers.__version__,'flash_attn':flash_attn.__version__,'training_started':False},indent=2))
PY
echo ready_for_official_smoke > "$OUT/stage"

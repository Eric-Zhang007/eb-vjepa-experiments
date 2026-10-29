#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
REPO=$ROOT/repos/RoboDojo
OUT=$ROOT/experiments/robodojo_setup
mkdir -p "$OUT" "$ROOT/cache/tmp"
exec 9>"$OUT/setup.lock"
flock -n 9 || exit 0
trap 'rc=$?; echo "$rc" > "$OUT/setup.exitcode"' EXIT
export TMPDIR="$ROOT/cache/tmp" HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 TERM=xterm-256color CUDA_HOME=/usr/local/cuda-12.8
source "$ROOT/envs/robodojo/bin/activate"
export PATH="$CUDA_HOME/bin:$PATH"
# Wait only for the exact bootstrap processes started during this session.
for pid in 959834 953697; do
  while [[ -r /proc/$pid/cmdline ]] && tr '\0' ' ' < /proc/$pid/cmdline | grep -Eq '(pip install torch==2.7.0|git .*clone --recurse-submodules.*RoboDojo)'; do sleep 15; done
done
cd "$REPO"
git -c http.proxy=http://127.0.0.1:7890 submodule update --init --recursive
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
cat > "$OUT/constraints.txt" <<'EOF'
numpy==1.26.0
packaging==23.0
typing_extensions==4.12.2
filelock==3.13.1
websockets==12.0
scipy==1.15.3
warp-lang==1.11.0
click==8.1.7
psutil==5.9.8
wheel==0.45.1
starlette==0.45.3
stable-baselines3<2.8
onnx>=1.18,<1.22
ipython<9
virtualenv==20.30.0
torch==2.7.0
torchvision==0.22.0
torchaudio==2.7.0
EOF
export PIP_CONSTRAINT="$OUT/constraints.txt"
python -m pip install -r scripts/requirements.txt opencv-python-headless==4.11.0.86 pillow matplotlib scikit-learn cmake ninja imageio-ffmpeg
# Pin runtime packages to upstream installer versions before installing simulator extensions.
python -m pip install numpy==1.26.0 packaging==23.0 typing_extensions==4.12.2 filelock==3.13.1 websockets==12.0 scipy==1.15.3 warp-lang==1.11.0 click==8.1.7 psutil==5.9.8 wheel==0.45.1 starlette==0.45.3 'onnx>=1.18,<1.22' 'ipython<9' virtualenv==20.30.0
python - <<'PY'
from pathlib import Path
import imageio_ffmpeg,sys
p=Path(sys.prefix)/'bin/ffmpeg'
if not p.exists(): p.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
PY
(cd third_party/IsaacLab && bash isaaclab.sh --install none)
python -m pip install -e 'third_party/curobo[cu12]' --no-build-isolation
python -m pip install -e XPolicyLab
python -m pip check > "$OUT/pip_check.txt"
python - <<'PY'
import torch,isaacsim,isaaclab
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
print('torch',torch.__version__,'cuda',torch.cuda.is_available())
PY
# Asset job runs independently. Do not report a render-ready environment prematurely.
while [[ ! -f "$OUT/assets.exitcode" ]]; do sleep 15; done
[[ "$(cat "$OUT/assets.exitcode")" == 0 ]]
bash scripts/robodojo.sh doctor --skip-conda --skip-isaac --skip-policy --summary "$OUT/doctor.json"
python "$ROOT/scripts/robodojo/render_smoke.py" > "$OUT/render_smoke.log" 2>&1
echo 'Completed install, doctor, and basic headless rendering smoke. Policy/task rollout still requires an adapter.'

#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
BENCH=${1:?robotwin or robodojo}; TASK=${2:?task name}; PORT=${3:-18081}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [[ "$BENCH" == robotwin ]]; then
  REPO=/data/users/jiachangzhang/RoboTwin
  export CUDA_VISIBLE_DEVICES="${SIM_GPU:-4}"
  source /data/users/jiachangzhang/.venv-robotwin/bin/activate
  cd "$REPO"
  exec bash scripts/eval_policy.sh --task_name "$TASK" --env_cfg_type "${ENV_CFG:-aloha_agilex}" --policy_name ELAF --host 127.0.0.1 --port "$PORT" --protocol ws --seed "${SEED:-0}" --test_num "${EPISODES:-1}" --num_workers "${WORKERS:-1}" --device_id 0 --eval_batch true
elif [[ "$BENCH" == robodojo ]]; then
  REPO=$ROOT/repos/RoboDojo
  source "$ROOT/envs/robodojo/bin/activate"
  cd "$REPO"
  export OMNI_KIT_ACCEPT_EULA=YES
  exec bash scripts/eval_policy.sh --root_dir "$REPO" --task_name "$TASK" --env_cfg_type "${ENV_CFG:-arx_x5}" --device_id "${SIM_GPU:-4}" --policy_name ELAF --host 127.0.0.1 --port "$PORT" --protocol ws --eval_batch true --seed "${SEED:-0}"
else
  echo 'Benchmark must be robotwin or robodojo' >&2; exit 2
fi

#!/usr/bin/env bash
# Launch a CPU-only XPolicy protocol bridge to the already-running batched model service.
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
XPL=${XPL_ROOT:?Set XPL_ROOT to selected benchmark/XPolicyLab}
PYTHON=${ELAF_BRIDGE_PYTHON:-/data/users/jiachangzhang/.venv-robotwin/bin/python}
CONFIG=${1:-$XPL/policy/ELAF/deploy.yml}
export PYTHONPATH="$ROOT:$(dirname "$XPL"):$XPL:${PYTHONPATH:-}"
exec "$PYTHON" "$XPL/setup_policy_server.py" --config_path "$CONFIG"

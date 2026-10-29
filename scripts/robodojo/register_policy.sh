#!/usr/bin/env bash
# Register lightweight adapter in one selected XPolicyLab checkout.
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
XPL=${1:?Usage: register_policy.sh /absolute/path/to/XPolicyLab}
[[ -f "$XPL/setup_policy_server.py" ]] || { echo 'Not an XPolicyLab checkout' >&2; exit 2; }
DEST="$XPL/policy/ELAF"
mkdir -p "$DEST"
cp "$ROOT/scripts/robodojo/elaf_policy/"* "$DEST/"
echo "Registered $DEST. Configure camera/state/action contract before launching."

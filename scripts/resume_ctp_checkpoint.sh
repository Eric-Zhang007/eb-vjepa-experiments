#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/users/jiachangzhang/eb_vjepa
CKPT="$ROOT/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
exec 9>"$ROOT/checkpoints/vjepa2_1/ctp_download.lock"
flock -n 9 || exit 0
if [ "$(stat -c %s "$CKPT")" = 5151198524 ]; then
    echo "Checkpoint already has expected size; tensor inspection still required."
    exit 0
fi
export TMPDIR="$ROOT/cache/tmp"
export https_proxy=http://127.0.0.1:7896
export http_proxy=http://127.0.0.1:7896
curl --fail --location --retry 8 --retry-delay 5 \
    --connect-timeout 30 --speed-limit 20000 --speed-time 120 \
    --continue-at - --output "$CKPT" \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
test "$(stat -c %s "$CKPT")" = 5151198524
echo "Download completed. Architecture/load inspection is pending."

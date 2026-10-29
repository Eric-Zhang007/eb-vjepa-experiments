#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/cache/imagemagick"
mkdir -p "$DEST/debs" "$DEST/root"
cd "$DEST/debs"
apt-get download libmagickwand-6.q16-6 libmagickcore-6.q16-6 imagemagick-6-common liblcms2-2 liblqr-1-0 libfftw3-double3
for package in ./*.deb; do
    dpkg-deb -x "$package" "$DEST/root"
done
export LD_LIBRARY_PATH="$DEST/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$DEST/home"
if [ ! -e "$DEST/home/lib" ]; then
    ln -s "$DEST/root/usr/lib/x86_64-linux-gnu" "$DEST/home/lib"
fi
ldd "$DEST/root/usr/lib/x86_64-linux-gnu/libMagickWand-6.Q16.so.6"

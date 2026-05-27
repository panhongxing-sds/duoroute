#!/usr/bin/env bash
# Download and extract DuoRoute data pack from GitHub Release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${DUOROUTE_DATA_TAG:-v0.1.0-data}"
REPO="${DUOROUTE_DATA_REPO:-panhongxing-sds/duoroute}"
ASSET="duoroute-data.tar.zst"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
DEST="${1:-$ROOT/../duoroute-data.tar.zst}"

echo "Downloading ${URL}"
curl -L --progress-bar -o "$DEST" "$URL"
echo "Extracting into ${ROOT} ..."
tar -C "$ROOT" -I zstd -xf "$DEST"
echo "Done. Run: python3 tests/test_duoroute.py"

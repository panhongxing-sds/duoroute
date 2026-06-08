#!/usr/bin/env bash
# Download and extract RegretRouter flagship data pack from GitHub Release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${REGRETROUTER_DATA_TAG:-v1.0.0-data}"
REPO="${REGRETROUTER_DATA_REPO:-panhongxing-sds/duoroute}"
ASSET="regretrouter_data_flagship.tar.zst"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"
DEST="${1:-$ROOT/data_pack/${ASSET}}"

if [[ -d "$ROOT/data/seed42_flagship/train" && -f "$ROOT/data/seed42_flagship/train/grouped.npz" ]]; then
  echo "Data already present at $ROOT/data/seed42_flagship — skipping download."
  exit 0
fi

mkdir -p "$(dirname "$DEST")" "$ROOT/data"
echo "Downloading ${URL}"
if ! curl -fL --progress-bar -o "$DEST" "$URL"; then
  echo "ERROR: download failed. Create Release ${TAG} with asset ${ASSET} (see docs/DATA_DOWNLOAD.md)." >&2
  exit 1
fi
echo "Extracting into ${ROOT} ..."
tar -C "$ROOT" -I zstd -xf "$DEST"
echo "Done. Run: python3 tests/test_duoroute.py"

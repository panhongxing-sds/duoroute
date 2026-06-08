#!/usr/bin/env bash
# One-command setup: install package + fetch flagship data if missing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Installing regretrouter (editable) ..."
pip install -e .

echo "==> Fetching data (Release tarball or local data_pack) ..."
if [[ -f "$ROOT/data_pack/regretrouter_data_flagship.tar.zst" ]]; then
  if [[ ! -d "$ROOT/data/seed42_flagship/train" ]]; then
    echo "Extracting bundled data_pack/regretrouter_data_flagship.tar.zst ..."
    mkdir -p "$ROOT/data"
    tar -C "$ROOT" -I zstd -xf "$ROOT/data_pack/regretrouter_data_flagship.tar.zst"
  else
    echo "data/seed42_flagship already present."
  fi
else
  bash "$ROOT/scripts/download_data.sh"
fi

echo "==> Running smoke tests ..."
python3 tests/test_duoroute.py

echo ""
echo "Setup complete. Quick reproduction:"
echo "  python3 scripts/run_multiseed_main_tables.py --quick"

#!/usr/bin/env bash
# Package seed42_flagship data for GitHub Release (compressed embeddings + meta.json.gz).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${REGRETROUTER_DATA_ROOT:-$ROOT/data}"
OUT="${1:-$ROOT/data_pack/regretrouter_data_flagship.tar.zst}"
POOL="seed42_flagship"
BASE="$DATA_ROOT/$POOL"

if [[ ! -d "$BASE/train" ]]; then
  echo "ERROR: $BASE not found. Set REGRETROUTER_DATA_ROOT or copy seed42_flagship into data/." >&2
  exit 1
fi

collect_pool() {
  local pool="$1"
  local base="$DATA_ROOT/$pool"
  for f in config.json model_cards.json embedding_meta.json \
           question_embeddings.f16.npz model_embeddings.f16.npz; do
    [[ -f "$base/$f" ]] && echo "data/$pool/$f"
  done
  [[ -f "$base/duoroute_config.yaml" ]] && echo "data/$pool/duoroute_config.yaml"
  for split in train val test; do
    for f in grouped.npz meta.json.gz; do
      [[ -f "$base/$split/$f" ]] && echo "data/$pool/$split/$f"
    done
    [[ -f "$base/$split/response_embeddings.f16.npz" ]] && \
      echo "data/$pool/$split/response_embeddings.f16.npz"
  done
}

mapfile -t FILES < <(collect_pool "$POOL")

TAR_ROOT="$(dirname "$DATA_ROOT")"
echo "Packing ${#FILES[@]} files from $BASE -> $OUT"
mkdir -p "$(dirname "$OUT")"
tar -C "$TAR_ROOT" -I 'zstd -T0 -3' -cf "$OUT" "${FILES[@]}"
ls -lh "$OUT"

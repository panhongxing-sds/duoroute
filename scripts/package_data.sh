#!/usr/bin/env bash
# Package reproducible data + checkpoints for GitHub Release upload.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/../duoroute-data.tar.zst}"

collect() {
  local pool="$1"
  local base="$ROOT/data/$pool"
  for f in config.json model_cards.json embedding_meta.json \
           question_embeddings.f16.npz model_embeddings.f16.npz; do
    [[ -f "$base/$f" ]] && echo "data/$pool/$f"
  done
  [[ -f "$base/duoroute_config.yaml" ]] && echo "data/$pool/duoroute_config.yaml"
  for split in train val test; do
    for f in grouped.npz meta.json.gz; do
      [[ -f "$base/$split/$f" ]] && echo "data/$pool/$split/$f"
    done
  done
  [[ -f "$base/train/response_embeddings.f16.npz" ]] && echo "data/$pool/train/response_embeddings.f16.npz"
}

mapfile -t FILES < <(
  collect seed42_small
  collect seed42_flagship
  echo outputs/checkpoints/seed42_small/best.pt
  echo outputs/checkpoints/seed42_flagship/best.pt
)

echo "Packing ${#FILES[@]} files -> $OUT"
tar -C "$ROOT" -I 'zstd -T0 -3' -cf "$OUT" "${FILES[@]}"
ls -lh "$OUT"

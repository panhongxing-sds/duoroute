#!/usr/bin/env bash
# Run after seed42_small finishes. Uses checkpointed API saves.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=data/seed42_flagship
CFG=configs/embedding.yaml

python3 scripts/build_embeddings.py --data-dir "$DATA" --target query --embedding-config "$CFG"
python3 scripts/build_embeddings.py --data-dir "$DATA" --target model --embedding-config "$CFG"
python3 scripts/build_embeddings.py \
  --data-dir "$DATA" \
  --target response \
  --response-splits train \
  --skip-empty-responses \
  --embedding-config "$CFG" \
  --checkpoint-every-batches 64

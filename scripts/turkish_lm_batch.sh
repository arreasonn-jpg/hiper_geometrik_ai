#!/bin/bash
# Turkish LM batch runner (virgüllü seed listesi).
# Kullanim: scripts/turkish_lm_batch.sh <baslangic> <bitis> <out_json>
set -euo pipefail
START="$1"
END="$2"
OUT="$3"
LOG="${OUT%.json}.log"
SEEDS=$(seq -s, "$START" "$END")
cd "$(dirname "$0")/.."
source .venv/bin/activate
echo "=== Batch: seeds=$SEEDS, out=$OUT ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
# Not: --signature-profile standard -> LM profili "full"
PYTHONUNBUFFERED=1 python -m hga turkce-lm \
    --signature-profile standard --seeds "$SEEDS" \
    --out "$OUT" --markdown "${OUT%.json}.md" \
    2>&1 | tee "$LOG"
echo "=== Bitti ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"

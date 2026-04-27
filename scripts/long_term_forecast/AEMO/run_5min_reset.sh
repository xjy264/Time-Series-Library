#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models=(
  Autoformer_5min_reset
  TimeMixer_5min_reset
)

for model in "${models[@]}"; do
  PRED_LENS="${PRED_LENS:-24}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" \
  PATIENCE="${PATIENCE:-3}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  DES="${DES:-AEMO-5min-reset}" \
  bash "${script_dir}/${model}.sh"
done

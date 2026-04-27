#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models=(
  DLinear_5min
  Informer_5min
  Autoformer_5min
  PatchTST_5min
  iTransformer_5min
  TimeMixer_5min
)

for model in "${models[@]}"; do
  PRED_LENS="${PRED_LENS:-24}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" \
  PATIENCE="${PATIENCE:-3}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  DES="${DES:-AEMO-5min}" \
  bash "${script_dir}/${model}.sh"
done

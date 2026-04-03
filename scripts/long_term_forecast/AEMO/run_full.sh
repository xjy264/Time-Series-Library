#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models=(DLinear TimeXer iTransformer PatchTST Informer Autoformer TimeMixer)

for model in "${models[@]}"; do
  PRED_LENS="${PRED_LENS:-24 48 96}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" \
  PATIENCE="${PATIENCE:-3}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  DES="${DES:-AEMO-full}" \
  bash "${script_dir}/${model}.sh"
done

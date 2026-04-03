#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models=(DLinear TimeXer iTransformer PatchTST Informer Autoformer TimeMixer)

for model in "${models[@]}"; do
  PRED_LENS="${PRED_LENS:-24}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}" \
  PATIENCE="${PATIENCE:-1}" \
  NUM_WORKERS="${NUM_WORKERS:-0}" \
  DES="${DES:-AEMO-smoke}" \
  bash "${script_dir}/${model}.sh"
done

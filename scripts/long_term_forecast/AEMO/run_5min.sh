#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
models=(
  DLinear_5min
  PatchTST_5min
  Informer_5min
  Autoformer_5min
  TimesNet_5min
  TimeXer_5min
  VPPGDFNet_5min
)

for model in "${models[@]}"; do
  PRED_LENS="${PRED_LENS:-24 48 96 288}" \
  SEQ_LEN="${SEQ_LEN:-288}" \
  LABEL_LEN="${LABEL_LEN:-144}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" \
  PATIENCE="${PATIENCE:-3}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  DES="${DES:-AEMO-5min}" \
  bash "${script_dir}/${model}.sh"
done

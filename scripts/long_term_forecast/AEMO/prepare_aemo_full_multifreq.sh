#!/usr/bin/env bash
set -euo pipefail

input_path="${AEMO_INPUT:-./data/aemo_vic1/aemo_vic1_dispatchis_vic1_5min_2022-08-25_2025-08-24.csv}"
output_dir="${OUTPUT_DIR:-./dataset/aemo_vic1}"
output_prefix="${OUTPUT_PREFIX:-aemo_vic1_dispatchis_vic1_full}"
freqs="${FREQS:-5min 15min 30min 1h}"

python3 -u tools/prepare_aemo_full_multifreq_dataset.py \
  --input "${input_path}" \
  --output-dir "${output_dir}" \
  --output-prefix "${output_prefix}" \
  --freqs ${freqs}

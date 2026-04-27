#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

cd "${repo_root}"

python3 -u tools/prepare_aemo_full_multifreq_dataset.py \
  --input "${AEMO_INPUT:-./data/aemo_vic1/aemo_vic1_dispatchis_vic1_5min_2022-08-25_2025-08-24.csv}" \
  --output-dir ./dataset/aemo_vic1 \
  --output-prefix aemo_vic1_dispatchis_vic1_full \
  --freqs 5min

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

bash "${script_dir}/TimeXer_5min.sh"

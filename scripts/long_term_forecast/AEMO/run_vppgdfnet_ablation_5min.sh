#!/usr/bin/env bash
set -u -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
cd "${repo_root}"

ablations=(
  full
  no_exog
  unified_exog
  no_final_gate
)
pred_lens=(24 48 96 288)

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "${value}"
}

extract_metric() {
  local log_file="$1"
  local key="$2"
  python3 - "$log_file" "$key" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
text = path.read_text(errors="ignore") if path.exists() else ""
matches = re.findall(r"mse:([-+0-9.eE]+), mae:([-+0-9.eE]+), dtw:([^\n\r]+)", text)
if not matches:
    sys.exit(0)
mse, mae, dtw = matches[-1]
print({"mse": mse, "mae": mae, "dtw": dtw.strip()}[key])
PY
}

run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${LOG_DIR:-results/aemo_vic1/vppgdfnet_ablation_5min_${run_id}}"
summary_csv="${SUMMARY_CSV:-${log_dir}/summary.csv}"
seq_len="${SEQ_LEN:-288}"
label_len="${LABEL_LEN:-144}"
train_epochs="${TRAIN_EPOCHS:-10}"
patience="${PATIENCE:-3}"
num_workers="${NUM_WORKERS:-4}"

mkdir -p "${log_dir}"
printf 'run_id,model,ablation,seq_len,pred_len,status,mse,mae,dtw,log_file,started_at,ended_at,exit_code\n' > "${summary_csv}"

for ablation in "${ablations[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    log_file="${log_dir}/vppgdfnet_${ablation}_sl${seq_len}_pl${pred_len}.log"
    started_at="$(date -Iseconds)"

    echo "[${started_at}] START VPPGDFNet ablation=${ablation} seq_len=${seq_len} pred_len=${pred_len}"

    set +e
    VPP_ABLATION="${ablation}" \
    PRED_LENS="${pred_len}" \
    SEQ_LEN="${seq_len}" \
    LABEL_LEN="${label_len}" \
    TRAIN_EPOCHS="${train_epochs}" \
    PATIENCE="${patience}" \
    NUM_WORKERS="${num_workers}" \
    DES="${DES:-AEMO-5min-vppgdfnet-ablation}" \
    bash "${script_dir}/VPPGDFNet_5min.sh" > "${log_file}" 2>&1
    exit_code=$?
    set -e

    ended_at="$(date -Iseconds)"
    if [[ ${exit_code} -eq 0 ]]; then
      status="success"
    else
      status="failed"
    fi

    mse="$(extract_metric "${log_file}" mse)"
    mae="$(extract_metric "${log_file}" mae)"
    dtw="$(extract_metric "${log_file}" dtw)"

    {
      csv_escape "${run_id}"; printf ','
      csv_escape "VPPGDFNet"; printf ','
      csv_escape "${ablation}"; printf ','
      csv_escape "${seq_len}"; printf ','
      csv_escape "${pred_len}"; printf ','
      csv_escape "${status}"; printf ','
      csv_escape "${mse}"; printf ','
      csv_escape "${mae}"; printf ','
      csv_escape "${dtw}"; printf ','
      csv_escape "${log_file}"; printf ','
      csv_escape "${started_at}"; printf ','
      csv_escape "${ended_at}"; printf ','
      csv_escape "${exit_code}"; printf '\n'
    } >> "${summary_csv}"

    echo "[${ended_at}] END VPPGDFNet ablation=${ablation} seq_len=${seq_len} pred_len=${pred_len} status=${status} exit_code=${exit_code} mse=${mse} mae=${mae}"
  done
done

echo "Wrote summary to ${summary_csv}"

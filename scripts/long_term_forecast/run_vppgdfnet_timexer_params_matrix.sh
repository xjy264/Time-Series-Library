#!/usr/bin/env bash
set -u -o pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${LOG_DIR:-results/general_vppgdfnet_timexer_params/${run_id}}"
summary_csv="${SUMMARY_CSV:-${log_dir}/summary.csv}"
averages_csv="${AVERAGES_CSV:-${log_dir}/averages.csv}"
mkdir -p "${log_dir}"

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

write_averages() {
  python3 - "$summary_csv" "$averages_csv" <<'PY'
import csv
import sys
from collections import OrderedDict
from pathlib import Path

summary_path = Path(sys.argv[1])
averages_path = Path(sys.argv[2])
rows = list(csv.DictReader(summary_path.open(newline="")))
order = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "Traffic"]
groups = OrderedDict((dataset, []) for dataset in order)
for row in rows:
    groups.setdefault(row["dataset"], []).append(row)

output_rows = []
all_success = []
for dataset, dataset_rows in groups.items():
    success_rows = [
        row for row in dataset_rows
        if row["status"] == "success" and row["mse"] and row["mae"]
    ]
    failed_count = sum(1 for row in dataset_rows if row["status"] != "success")
    all_success.extend(success_rows)
    if success_rows:
        avg_mse = sum(float(row["mse"]) for row in success_rows) / len(success_rows)
        avg_mae = sum(float(row["mae"]) for row in success_rows) / len(success_rows)
        avg_mse_text = f"{avg_mse:.10f}"
        avg_mae_text = f"{avg_mae:.10f}"
    else:
        avg_mse_text = ""
        avg_mae_text = ""
    output_rows.append({
        "dataset": dataset,
        "avg_mse": avg_mse_text,
        "avg_mae": avg_mae_text,
        "success_count": str(len(success_rows)),
        "failed_count": str(failed_count),
    })

if all_success:
    all_avg_mse = sum(float(row["mse"]) for row in all_success) / len(all_success)
    all_avg_mae = sum(float(row["mae"]) for row in all_success) / len(all_success)
    all_avg_mse_text = f"{all_avg_mse:.10f}"
    all_avg_mae_text = f"{all_avg_mae:.10f}"
else:
    all_avg_mse_text = ""
    all_avg_mae_text = ""
output_rows.append({
    "dataset": "ALL",
    "avg_mse": all_avg_mse_text,
    "avg_mae": all_avg_mae_text,
    "success_count": str(len(all_success)),
    "failed_count": str(sum(1 for row in rows if row["status"] != "success")),
})

with averages_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["dataset", "avg_mse", "avg_mae", "success_count", "failed_count"])
    writer.writeheader()
    writer.writerows(output_rows)
PY
}

printf 'run_id,dataset,model,seq_len,label_len,pred_len,features,mse,mae,status,log_file,started_at,ended_at,exit_code\n' > "${summary_csv}"

datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL Traffic)
pred_lens=(96 192 336 720)

script_for_dataset() {
  case "$1" in
    ETTh1) echo "scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTh1.sh" ;;
    ETTh2) echo "scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTh2.sh" ;;
    ETTm1) echo "scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTm1.sh" ;;
    ETTm2) echo "scripts/long_term_forecast/ETT_script/VPPGDFNet_ETTm2.sh" ;;
    Weather) echo "scripts/long_term_forecast/Weather_script/VPPGDFNet.sh" ;;
    ECL) echo "scripts/long_term_forecast/ECL_script/VPPGDFNet.sh" ;;
    Traffic) echo "scripts/long_term_forecast/Traffic_script/VPPGDFNet.sh" ;;
    *) return 1 ;;
  esac
}

slug_for_dataset() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

for dataset in "${datasets[@]}"; do
  script_path="$(script_for_dataset "${dataset}")"
  dataset_slug="$(slug_for_dataset "${dataset}")"
  for pred_len in "${pred_lens[@]}"; do
    log_file="${log_dir}/${dataset_slug}_pl${pred_len}.log"
    started_at="$(date -Iseconds)"
    echo "[${started_at}] START VPPGDFNet dataset=${dataset} pred_len=${pred_len}"

    set +e
    PRED_LENS="${pred_len}" bash "${script_path}" > "${log_file}" 2>&1
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

    {
      csv_escape "${run_id}"; printf ','
      csv_escape "${dataset}"; printf ','
      csv_escape "VPPGDFNet"; printf ','
      csv_escape "96"; printf ','
      csv_escape "48"; printf ','
      csv_escape "${pred_len}"; printf ','
      csv_escape "M"; printf ','
      csv_escape "${mse}"; printf ','
      csv_escape "${mae}"; printf ','
      csv_escape "${status}"; printf ','
      csv_escape "${log_file}"; printf ','
      csv_escape "${started_at}"; printf ','
      csv_escape "${ended_at}"; printf ','
      csv_escape "${exit_code}"; printf '\n'
    } >> "${summary_csv}"

    write_averages
    echo "[${ended_at}] END VPPGDFNet dataset=${dataset} pred_len=${pred_len} status=${status} exit_code=${exit_code} mse=${mse} mae=${mae}"
  done
done

write_averages
echo "Wrote summary to ${summary_csv}"
echo "Wrote averages to ${averages_csv}"

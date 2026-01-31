#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_v_stitched.sh ETTh1 V1
#   bash scripts/run_v_stitched.sh ETTm1 V10

DATASET_NAME="$1"   # ETTh1 or ETTm1
MODEL_NAME="$2"     # V1..V10

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$ROOT_DIR/dataset/ETT-small"
DATA_PATH="$DATASET_NAME.csv"

RUN_DIR="$ROOT_DIR/records/runs/${DATASET_NAME}/${MODEL_NAME}"
mkdir -p "$RUN_DIR"

# You can adjust these defaults if needed.
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96

ENC_IN=7
DEC_IN=7
C_OUT=7

# Common model hyperparams (reasonable defaults for ETT)
D_MODEL=512
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=2048
MOVING_AVG=25
TOP_K=5
NUM_KERNELS=6

# Training defaults
TRAIN_EPOCHS=1
BATCH_SIZE=32
LR=0.0001
NUM_WORKERS=2

CMD=(
  python -u "$ROOT_DIR/run.py"
  --task_name long_term_forecast
  --is_training 1
  --use_gpu False
  --gpu_type mps
  --root_path "$DATA_ROOT/"
  --data_path "$DATA_PATH"
  --model_id "${DATASET_NAME}_${MODEL_NAME}_sl${SEQ_LEN}_pl${PRED_LEN}"
  --model "$MODEL_NAME"
  --data "$DATASET_NAME"
  --features M
  --seq_len "$SEQ_LEN"
  --label_len "$LABEL_LEN"
  --pred_len "$PRED_LEN"
  --enc_in "$ENC_IN"
  --dec_in "$DEC_IN"
  --c_out "$C_OUT"
  --d_model "$D_MODEL"
  --n_heads "$N_HEADS"
  --e_layers "$E_LAYERS"
  --d_layers "$D_LAYERS"
  --d_ff "$D_FF"
  --moving_avg "$MOVING_AVG"
  --top_k "$TOP_K"
  --num_kernels "$NUM_KERNELS"
  --train_epochs "$TRAIN_EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LR"
  --num_workers "$NUM_WORKERS"
  --des "V_stitched"
)

printf "Running: %q " "${CMD[@]}" | tee "$RUN_DIR/cmd.txt"
echo | tee -a "$RUN_DIR/cmd.txt"

# Save environment snapshot
python -V > "$RUN_DIR/python.txt" 2>&1 || true
pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true

# Run and tee logs
"${CMD[@]}" 2>&1 | tee "$RUN_DIR/run.log"

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=TimeXer
pred_lens="${PRED_LENS:-24}"
train_epochs="${TRAIN_EPOCHS:-10}"
patience="${PATIENCE:-3}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
learning_rate="${LEARNING_RATE:-0.0005}"
des="${DES:-AEMO-full-1h}"
patch_len="${PATCH_LEN:-1}"
seq_len="${SEQ_LEN:-168}"
label_len="${LABEL_LEN:-48}"
data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_1h.csv}"
freq="${FREQ:-h}"
enc_in="${ENC_IN:-12}"
dec_in="${DEC_IN:-12}"
d_model="${D_MODEL:-128}"
d_ff="${D_FF:-256}"

for pred_len in ${pred_lens}; do
  python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/aemo_vic1/ \
    --data_path "${data_path}" \
    --model_id aemo_timexer_full_1h_${seq_len}_${pred_len} \
    --model ${model_name} \
    --data custom \
    --features MS \
    --target net_load \
    --freq "${freq}" \
    --seq_len "${seq_len}" \
    --label_len "${label_len}" \
    --pred_len "${pred_len}" \
    --e_layers 1 \
    --factor 3 \
    --enc_in "${enc_in}" \
    --dec_in "${dec_in}" \
    --c_out 1 \
    --des "${des}" \
    --patch_len "${patch_len}" \
    --d_model "${d_model}" \
    --d_ff "${d_ff}" \
    --batch_size "${batch_size}" \
    --train_epochs "${train_epochs}" \
    --patience "${patience}" \
    --learning_rate "${learning_rate}" \
    --num_workers "${num_workers}" \
    --itr 1
done

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=PatchTST
pred_lens="${PRED_LENS:-24}"
train_epochs="${TRAIN_EPOCHS:-10}"
patience="${PATIENCE:-3}"
batch_size="${BATCH_SIZE:-32}"
num_workers="${NUM_WORKERS:-4}"
learning_rate="${LEARNING_RATE:-0.0005}"
des="${DES:-AEMO-5min}"
seq_len="${SEQ_LEN:-2016}"
label_len="${LABEL_LEN:-144}"
data_path="${DATA_PATH:-aemo_vic1_dispatchis_vic1_full_5min.csv}"
freq="${FREQ:-5min}"
enc_in="${ENC_IN:-12}"
dec_in="${DEC_IN:-12}"

for pred_len in ${pred_lens}; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/aemo_vic1/ \
    --data_path "${data_path}" \
    --model_id aemo_patchtst_5min_${seq_len}_${pred_len} \
    --model ${model_name} \
    --data custom \
    --features MS \
    --target net_load \
    --freq "${freq}" \
    --seq_len "${seq_len}" \
    --label_len "${label_len}" \
    --pred_len "${pred_len}" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in "${enc_in}" \
    --dec_in "${dec_in}" \
    --c_out 1 \
    --des "${des}" \
    --d_model 128 \
    --d_ff 256 \
    --batch_size "${batch_size}" \
    --train_epochs "${train_epochs}" \
    --patience "${patience}" \
    --learning_rate "${learning_rate}" \
    --num_workers "${num_workers}" \
    --itr 1
done

#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=TimeMixer
pred_lens="${PRED_LENS:-24 48 96}"
train_epochs="${TRAIN_EPOCHS:-20}"
patience="${PATIENCE:-5}"
batch_size="${BATCH_SIZE:-32}"
num_workers="${NUM_WORKERS:-4}"
learning_rate="${LEARNING_RATE:-0.001}"
des="${DES:-AEMO-full}"

for pred_len in ${pred_lens}; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/aemo_vic1/ \
    --data_path aemo_vic1_timexer_ms.csv \
    --model_id aemo_timemixer_168_${pred_len} \
    --model ${model_name} \
    --data custom \
    --features MS \
    --target net_load \
    --freq h \
    --seq_len 168 \
    --label_len 0 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 1 \
    --des ${des} \
    --d_model 32 \
    --d_ff 64 \
    --batch_size ${batch_size} \
    --train_epochs ${train_epochs} \
    --patience ${patience} \
    --learning_rate ${learning_rate} \
    --num_workers ${num_workers} \
    --down_sampling_layers 2 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --itr 1
done

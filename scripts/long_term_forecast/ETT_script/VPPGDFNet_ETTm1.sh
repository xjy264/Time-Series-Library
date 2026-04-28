set -e
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=VPPGDFNet
pred_lens="${PRED_LENS:-96 192 336 720}"

if [[ " ${pred_lens} " == *" 96 "* ]]; then
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id vppgdfnet_ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --batch_size 4 \
    --des 'VPPGDFNet-TimeXerParams' \
    --itr 1

fi

if [[ " ${pred_lens} " == *" 192 "* ]]; then
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id vppgdfnet_ETTm1_96_192 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 256 \
    --batch_size 4 \
    --des 'VPPGDFNet-TimeXerParams' \
    --itr 1

fi

if [[ " ${pred_lens} " == *" 336 "* ]]; then
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id vppgdfnet_ETTm1_96_336 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 1024 \
    --batch_size 4 \
    --des 'VPPGDFNet-TimeXerParams' \
    --itr 1

fi

if [[ " ${pred_lens} " == *" 720 "* ]]; then
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id vppgdfnet_ETTm1_96_720 \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 512 \
    --batch_size 4 \
    --des 'VPPGDFNet-TimeXerParams' \
    --itr 1
fi

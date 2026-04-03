model_name=TimeXer
des='Timexer-MS-AEMO'
patch_len=24

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/aemo_vic1/ \
  --data_path aemo_vic1_timexer_ms.csv \
  --model_id aemo_vic1_168_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --target net_load \
  --freq h \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 6 \
  --dec_in 6 \
  --c_out 1 \
  --des $des \
  --patch_len $patch_len \
  --d_model 128 \
  --d_ff 256 \
  --batch_size 16 \
  --itr 1

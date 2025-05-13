export CUDA_VISIBLE_DEVICES=1
model_name=SoloLinear
des='SoloLinear-N'


# ETTm1_96_96
python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_96 \
    --model $model_name \
    --data ETTm1 \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --e_layers 2 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 1 \
    --des ${des}_${init_type}_std${std_val} \
    --cycle True \
    --cycle_len 24 \
    --itr 1

# ETTm1_96_192
python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_192 \
    --model $model_name \
    --data ETTm1 \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 192 \
    --e_layers 2 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 1 \
    --des ${des}_${init_type}_std${std_val} \
    --cycle True \
    --cycle_len 24 \
    --itr 1

# ETTm1_96_336
python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_336 \
    --model $model_name \
    --data ETTm1 \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 336 \
    --e_layers 5 \
    --factor 6 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 1 \
    --des ${des}_${init_type}_std${std_val} \
    --cycle True \
    --cycle_len 24 \
    --itr 1

# ETTm1_96_720
python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_720 \
    --model $model_name \
    --data ETTm1 \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 720 \
    --e_layers 2 \
    --factor 3 \
    --enc_in 6 \
    --dec_in 6 \
    --c_out 1 \
    --des ${des}_${init_type}_std${std_val} \
    --cycle True \
    --cycle_len 24 \
    --itr 1

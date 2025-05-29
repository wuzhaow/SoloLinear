export CUDA_VISIBLE_DEVICES=1
model_name=SoloLinear
des='SoloLinear-N'

python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_96 \
    --model $model_name \
    --data custom \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --e_layers 2 \
    --factor 3 \
    --enc_in 320 \
    --dec_in 320 \
    --c_out 1 \
    --des ${des} \
    --cycle True \
    --cycle_len 24 \
    --batch_size 4 \
    --init_type normal \
    --itr 1

# ECL_96_192
python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_192 \
    --model $model_name \
    --data custom \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 192 \
    --e_layers 2 \
    --factor 3 \
    --enc_in 320 \
    --dec_in 320 \
    --c_out 1 \
    --des ${des} \
    --cycle True \
    --cycle_len 24 \
    --batch_size 4 \
    --init_type normal \
    --itr 1

# ECL_96_336
python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_336 \
    --model $model_name \
    --data custom \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 336 \
    --e_layers 2 \
    --factor 1 \
    --enc_in 320 \
    --dec_in 320 \
    --c_out 1 \
    --des ${des} \
    --cycle True \
    --cycle_len 24 \
    --batch_size 4 \
    --init_type normal \
    --itr 1

# ECL_96_720
python3 -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_720 \
    --model $model_name \
    --data custom \
    --features N \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 720 \
    --e_layers 2 \
    --factor 3 \
    --enc_in 320 \
    --dec_in 320 \
    --c_out 1 \
    --des ${des} \
    --cycle True \
    --batch_size 4 \
    --cycle_len 24 \
    --init_type normal \
    --itr 1
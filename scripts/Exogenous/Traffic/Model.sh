export CUDA_VISIBLE_DEVICES=1
model_name=Model
des='Model-N'
# Fixed initialization type and std value
init_type="normal"
std_val="0.01"
# Array of learning rates to search
learning_rates=("1e-2" "5e-3" "1e-3" "5e-4" "1e-4")
learning_rates=("1e-3")
for lr in "${learning_rates[@]}"; do
    echo "Running with init_type=$init_type, std=$std_val, learning_rate=$lr"
    # traffic_96_96
    python3 -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id traffic_96_96 \
        --model $model_name \
        --data custom \
        --features N \
        --seq_len 96 \
        --label_len 0 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 1 \
        --enc_in 861 \
        --dec_in 861 \
        --c_out 1 \
        --des ${des}_${init_type}_std${std_val}_lr${lr} \
        --cycle True \
        --cycle_len 24 \
        --batch_size 4 \
        --init_type $init_type \
        --mean 0.0 \
        --std $std_val \
        --learning_rate $lr \
        --itr 1
    # traffic_96_192
    python3 -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id traffic_96_192 \
        --model $model_name \
        --data custom \
        --features N \
        --seq_len 96 \
        --label_len 0 \
        --pred_len 192 \
        --e_layers 1 \
        --d_layers 1 \
        --factor 1 \
        --enc_in 861 \
        --dec_in 861 \
        --c_out 1 \
        --des ${des}_${init_type}_std${std_val}_lr${lr} \
        --cycle True \
        --cycle_len 24 \
        --batch_size 4 \
        --init_type $init_type \
        --mean 0.0 \
        --std $std_val \
        --learning_rate $lr \
        --itr 1
    # traffic_96_336
    python3 -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id traffic_96_336 \
        --model $model_name \
        --data custom \
        --features N \
        --seq_len 96 \
        --label_len 0 \
        --pred_len 336 \
        --e_layers 1 \
        --d_layers 1 \
        --factor 1 \
        --enc_in 861 \
        --dec_in 861 \
        --c_out 1 \
        --des ${des}_${init_type}_std${std_val}_lr${lr} \
        --cycle True \
        --cycle_len 24 \
        --batch_size 4 \
        --init_type $init_type \
        --mean 0.0 \
        --std $std_val \
        --learning_rate $lr \
        --itr 1
    # traffic_96_720
    python3 -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/traffic/ \
        --data_path traffic.csv \
        --model_id traffic_96_720 \
        --model $model_name \
        --data custom \
        --features N \
        --seq_len 96 \
        --label_len 0 \
        --pred_len 720 \
        --e_layers 1 \
        --d_layers 1 \
        --factor 1 \
        --enc_in 861 \
        --dec_in 861 \
        --c_out 1 \
        --des ${des}_${init_type}_std${std_val}_lr${lr} \
        --cycle True \
        --cycle_len 24 \
        --batch_size 4 \
        --init_type $init_type \
        --mean 0.0 \
        --std $std_val \
        --learning_rate $lr \
        --itr 1
done
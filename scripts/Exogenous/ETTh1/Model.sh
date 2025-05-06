export CUDA_VISIBLE_DEVICES=1
model_name=Model
des='Model-N'
# Array of initialization types and std values to search
init_types=("normal")
std_values=(1 0.1 0.01 0.001 0.0001)
#std_values=(0.00)

for init_type in "${init_types[@]}"; do
  # Set std_values based on init_type
    if [ "$init_type" = "zero" ]; then
        std_values=(0.00)
    else
        std_values=(1 0.1 0.01 0.001 0.0001)
    fi
    for std_val in "${std_values[@]}"; do
        echo "Running with init_type=$init_type, std=$std_val"

        # ETTh1_96_96
        python3 -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_96_96 \
            --model $model_name \
            --data ETTh1 \
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
            --init_type $init_type \
            --mean 0.0 \
            --std $std_val \
            --itr 1

        # ETTh1_96_192
        python3 -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_96_192 \
            --model $model_name \
            --data ETTh1 \
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
            --init_type $init_type \
            --mean 0.0 \
            --std $std_val \
            --itr 1

        # ETTh1_96_336
        python3 -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_96_336 \
            --model $model_name \
            --data ETTh1 \
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
            --init_type $init_type \
            --mean 0.0 \
            --std $std_val \
            --itr 1

        # ETTh1_96_720
        python3 -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_96_720 \
            --model $model_name \
            --data ETTh1 \
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
            --init_type $init_type \
            --mean 0.0 \
            --std $std_val \
            --itr 1
    done
done
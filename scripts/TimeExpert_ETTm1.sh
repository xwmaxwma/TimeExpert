export CUDA_VISIBLE_DEVICES=0

model_name=TimeExpert
setting_name="ETTm1_96_96"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id $setting_name \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 1 \
    --e_layers 2 \
    --batch_size 32 \
    --d_model 512 \
    --topk 2 \
    --shared



setting_name="ETTm1_96_192"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id $setting_name \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 2 \
    --itr 1 \
    --e_layers 4 \
    --batch_size 16 \
    --d_model 512 \
    --topk 4 \
    --shared


setting_name="ETTm1_96_336"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id $setting_name \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 4 \
    --itr 1 \
    --e_layers 2 \
    --batch_size 16 \
    --d_model 512 \
    --topk 8 
    

setting_name="ETTm1_96_720"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id $setting_name \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --n_heads 4 \
    --itr 1 \
    --e_layers 2 \
    --batch_size 128 \
    --d_model 768 \
    --topk 6
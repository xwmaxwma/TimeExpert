export CUDA_VISIBLE_DEVICES=0

model_name=TimeExpert
setting_name="Weather_96_96"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id $setting_name \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --patch_len 6 \
    --stride  3 \
    --learning_rate 0.00003 \
    --n_heads 32 \
    --dropout 0.0 \
    --itr 1 \
    --e_layers 3 \
    --batch_size 16 \
    --d_model 512 \
    --topk 6 



setting_name="Weather_96_192"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id $setting_name \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --n_heads 16 \
    --itr 1 \
    --e_layers 3 \
    --batch_size 32 \
    --d_model 512 \
    --topk 10 \
    --shared 


setting_name="Weather_96_336"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id $setting_name \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --patch_len 6 \
    --stride 3 \
    --learning_rate 0.00003 \
    --n_heads 8 \
    --itr 1 \
    --e_layers 3 \
    --batch_size 32 \
    --d_model 512 \
    --topk 6 
    

setting_name="Weather_96_720"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id $setting_name \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --patch_len 6 \
    --stride 3 \
    --learning_rate 0.00003 \
    --n_heads 16 \
    --itr 1 \
    --e_layers 3\
    --batch_size 32 \
    --d_model 512 \
    --topk 10 \
    --shared



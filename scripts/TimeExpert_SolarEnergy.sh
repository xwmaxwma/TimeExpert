export CUDA_VISIBLE_DEVICES=0

model_name=TimeExpert
setting_name="SloarEner_96_96"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id $setting_name \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --patch_len 8 \
    --stride  4 \
    --learning_rate 0.0002 \
    --n_heads 16 \
    --dropout 0.0 \
    --itr 1 \
    --e_layers 2 \
    --batch_size 16 \
    --d_model 512 \
    --topk 8 \
    --shared



setting_name="SloarEner_96_192"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id $setting_name \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --patch_len 8 \
    --stride  4 \
    --learning_rate 0.0003 \
    --n_heads 16 \
    --dropout 0.0 \
    --itr 1 \
    --e_layers 2 \
    --batch_size 16 \
    --d_model 512 \
    --topk 8 \
    --shared


setting_name="SloarEner_96_336"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id $setting_name \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --patch_len 8 \
    --stride  4 \
    --learning_rate 0.0002 \
    --n_heads 8 \
    --dropout 0.0 \
    --itr 1 \
    --e_layers 2 \
    --batch_size 8 \
    --d_model 512 \
    --topk 8 \
    --shared
    

setting_name="SloarEner_96_720"

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id $setting_name \
    --model $model_name \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --d_layers 1 \
    --factor 1 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --patch_len 8 \
    --stride  4 \
    --learning_rate 0.0003 \
    --n_heads 16 \
    --dropout 0.0 \
    --itr 1 \
    --e_layers 2 \
    --batch_size 16 \
    --d_model 512 \
    --topk 8 \
    --shared



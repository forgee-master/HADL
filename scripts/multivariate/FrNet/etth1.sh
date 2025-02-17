#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=FrNet
seq_len=512


root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for pred_len in 96
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type Linear \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48 72\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.0003
done

for pred_len in 192
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type Linear \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48\
      --emb 64\
      --itr 1 --batch_size 128 --learning_rate 0.0005
done

for pred_len in 336
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type Linear \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'truncation'\
      --aggregation_type 'avg'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 24 48\
      --emb 64\
      --itr 1 --batch_size 128 --learning_rate 0.0005
done

for pred_len in 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --train_type Linear \
      --enc_in 7 \
      --e_layers 1 \
      --n_heads 1 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.05\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 24\
      --stride 24\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'truncation'\
      --aggregation_type 'linear'\
      --channel_attention 0\
      --global_freq_pred 1\
      --period_list 24 48 72\
      --emb 96\
      --itr 1 --batch_size 128 --learning_rate 0.0005 
done
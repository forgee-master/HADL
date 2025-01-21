#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=512
model_name=FrNet

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --train_type Linear --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 2 \
      --n_heads 8 \
      --d_model 64 \
      --d_ff 128 \
      --dropout 0.1\
      --fc_dropout 0.1\
      --head_dropout 0.1\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --patience 10\
      --kernel_size 25\
      --lradj type3\
      --pred_head_type 'linear'\
      --aggregation_type 'linear'\
      --channel_attention 0\
      --global_freq_pred 0\
      --period_list 144 72\
      --emb 164\
      --decomposition 0\
      --itr 1 --batch_size 32 --learning_rate 0.0003 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done

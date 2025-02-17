#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=DLinear

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom


for seq_len in 512
do
for pred_len in 96 192 336 720
do    
    python -u run_longExp.py \
      --is_training 1 \
      --individual 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --train_epochs 50 \
      --patience 10 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done


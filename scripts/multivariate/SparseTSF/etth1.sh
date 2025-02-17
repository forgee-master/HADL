#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SparseTSF

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for seq_len in 512
do
for pred_len in 96 192 336 720
do    
    python -u run_longExp.py \
      --is_training 1 \
      --individual 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len 16 \
      --enc_in 7 \
      --train_epochs 100 \
      --patience 20 \
      --des 'Exp' \
      --model_type 'linear' \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

#model_type 'mlp'
#d_model 128
#
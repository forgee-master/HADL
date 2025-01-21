#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=SparseTSF

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

for noise_std in 0.0 0.3 0.7 1.3 1.7 2.3 
do
for seq_len in 512
do
for pred_len in 192
do    
    python -u run_noisetesting.py \
      --is_training 1 --noise_std  $noise_std \
      --individual 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M --train_type Linear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len 16 \
      --enc_in 7 \
      --train_epochs 50 \
      --patience 10 \
      --des 'Exp' \
      --model_type 'linear' \
      --itr 1 --batch_size 32 --learning_rate 0.01
done
done

#model_type 'mlp'
#d_model 128
#
done

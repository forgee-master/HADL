#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=HADL

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

rank=50

for noise_std in 0.0 0.3 0.7 1.3 1.7 2.3 
do
for seq_len in 512
do
for pred_len in 192
do    
    python -u run_noisetesting.py \
      --is_training 1 \
      --noise_std  $noise_std \
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
      --enc_in 7 \
      --train_epochs 50 \
      --rank $rank \
      --bias 1 \
      --enable_Haar 1 \
      --enable_DCT 1 \
      --enable_lowrank 1 \
      --enable_iDCT 0 \
      --patience 10 \
      --des 'Exp' \
      --regularizer 1 \
      --regularization_rate 0.1 \
      --itr 1 \
      --batch_size 32 \
      --learning_rate 0.01
done
done
done

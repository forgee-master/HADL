#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=ModernTCN

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
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --train_epochs 50 \
      --patience 10 \
      --train_type TCN \
      --ffn_ratio 8 \
      --patch_size 8 \
      --patch_stride 4 \
      --num_blocks 3 \
      --large_size 51 \
      --small_size 5 \
      --dims 64 64 64 64 \
      --head_dropout 0.0 \
      --dropout 0.3 \
      --lradj type3 \
      --use_multi_scale False \
      --small_kernel_merged False \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.0001
done
done
done

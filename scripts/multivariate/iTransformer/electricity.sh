#!/bin/sh

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=iTransformer

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

for seq_len in 512
do
for pred_len in 96 192 336 720
do    
    if [ $pred_len -eq 96 ] || [ $pred_len -eq 192 ]; then
      d_model=256
      d_ff=256
    else
      d_model=512
      d_ff=512
    fi
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --train_type nonlinear \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --d_model $d_model \
      --d_ff $d_ff \
      --train_epochs 100 \
      --patience 20 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.0001
done
done


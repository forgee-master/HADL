#!/bin/sh

if [ ! -d "./logs/FITS_fix/etth1_abl" ]; then
    mkdir ./logs/FITS_fix/etth1_abl
fi

model_name=FITS
H_order=12
seq_len=512
m=1

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
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M --train_type FITS \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --train_epochs 50 \
    --des 'Exp' \
    --train_mode $m \
    --H_order $H_order \
    --base_T 144 \
    --patience 10 \
    --itr 1 --batch_size 32 --learning_rate 0.0005
done
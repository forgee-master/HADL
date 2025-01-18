if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=TSMixer

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

for noise_std in 0.0 0.1 0.3 0.5 0.7 
do
for seq_len in 512
do
for pred_len in 96 192 336 720
do    
    python -u run_noisetesting.py \
      --is_training 1 \
      --individual 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --period_len 24 \
      --enc_in 7 \
      --norm_type "batch" \
      --num_blocks 2 \
      --ff_dim 64 \
      --normalize_before True \
      --train_epochs 1 \
      --patience 1 \
      --des 'Exp' \
      --itr 1 --batch_size 32 --learning_rate 0.02
done
done

#model_type 'mlp'
#d_model 128
#
done

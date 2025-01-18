import os
import subprocess

# Create logs directory if it doesn't exist
if not os.path.exists('./logs'):
    os.makedirs('./logs')

# Model and data configuration
model_name = 'FrNet'
seq_len = 512

root_path_name = './dataset/'
data_path_name = 'ETTh1.csv'
model_id_name = 'ETTh1'
data_name = 'ETTh1'

# Prediction length
pred_len = 96

# Construct command arguments
command = [
    'python', '-u', 'run_longExp.py',
    '--is_training', '1',
    '--root_path', root_path_name,
    '--data_path', data_path_name,
    '--model_id', f'{model_id_name}_{seq_len}_{pred_len}',
    '--model', model_name,
    '--data', data_name,
    '--features', 'M',
    '--seq_len', str(seq_len),
    '--pred_len', str(pred_len),
    '--train_type', 'Linear',
    '--enc_in', '7',
    '--e_layers', '1',
    '--n_heads', '1',
    '--d_model', '16',
    '--d_ff', '128',
    '--dropout', '0.05',
    '--fc_dropout', '0.1',
    '--head_dropout', '0.1',
    '--patch_len', '24',
    '--stride', '24',
    '--des', 'Exp',
    '--train_epochs', '100',
    '--patience', '10',
    '--kernel_size', '25',
    '--lradj', 'type4',
    '--pred_head_type', 'truncation',
    '--aggregation_type', 'avg',
    '--channel_attention', '0',
    '--global_freq_pred', '0',
    '--period_list', '24', '48', '72',
    '--emb', '96',
    '--itr', '1', 
    '--batch_size', '128', 
    '--learning_rate', '0.0003'
]

# Execute the command
subprocess.run(command, check=True)

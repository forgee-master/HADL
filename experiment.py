import os
import subprocess

# Ensure the logs directory exists
if not os.path.exists("./logs"):
    os.makedirs("./logs")

# Define parameters
model_name = "HADL"
root_path_name = "./dataset/"
data_path_name = "ETTh1.csv"
model_id_name = "ETTh1"
data_name = "ETTh1"
rank = 50

# Loop through sequence and prediction lengths
for seq_len in [512]:
    for pred_len in [96]:
        model_id = f"{model_id_name}_{seq_len}_{pred_len}"
        
        # Command construction
        command = [
            "python", "-u", "run_longExp.py",
            "--is_training", "1",
            "--individual", "0",
            "--root_path", root_path_name,
            "--data_path", data_path_name,
            "--model_id", model_id,
            "--model", model_name,
            "--data", data_name,
            "--features", "M",
            "--train_type", "Linear",
            "--seq_len", str(seq_len),
            "--pred_len", str(pred_len),
            "--enc_in", "7",
            "--train_epochs", "50",
            "--rank", str(rank),
            "--bias", "1",
            "--enable_Haar", "1",
            "--enable_DCT", "1",
            "--enable_lowrank", "1",
            "--enable_iDCT", "0",
            "--patience", "10",
            "--des", "Exp",
            "--regularizer", "1",
            "--regularization_rate", "0.1",
            "--itr", "1",
            "--batch_size", "32",
            "--learning_rate", "0.01"
        ]
        
        # Run command
        subprocess.run(command)
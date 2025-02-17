import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.fft import dct, idct
import math

class DiscreteCosineTransform(Function):
    """
    Implements a custom autograd function for Discrete Cosine Transform (DCT).
    """
    @staticmethod
    def forward(ctx, input):
        # Convert PyTorch tensor to NumPy array for scipy operations
        input_np = input.cpu().numpy()
        
        # Apply DCT (Type-II) with orthonormalization
        transformed_np = dct(input_np, type=2, norm="ortho", axis=-1)
        
        # Convert back to PyTorch tensor and return
        output = torch.from_numpy(transformed_np).to(input.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Convert gradient to NumPy array
        grad_output_np = grad_output.cpu().numpy()
        
        # Apply IDCT (Type-II) with orthonormalization
        grad_input_np = idct(grad_output_np, type=2, norm='ortho', axis=-1)
        
        # Convert back to PyTorch tensor and return
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        return grad_input
    
class iDiscreteCosineTransform(Function):
    """
    Implements a custom autograd function for Inverse Discrete Cosine Transform (iDCT).
    """
    @staticmethod
    def forward(ctx, input):
        # Convert PyTorch tensor to NumPy array
        input_np = input.cpu().numpy()

        # Apply IDCT using scipy
        transformed_np = idct(input_np, type=2, axis=-1)

        # Convert back to PyTorch tensor
        output = torch.from_numpy(transformed_np).to(input.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Convert gradient to NumPy array
        grad_output_np = grad_output.cpu().numpy()

        # Apply DCT using scipy
        grad_input_np = dct(grad_output_np, type=2, axis=-1)
        
        # Convert back to PyTorch tensor
        grad_input = torch.from_numpy(grad_input_np).to(grad_output.device)
        return grad_input

class LowRank(nn.Module):
    """
    Implements a low-rank approximation layer using two smaller weight matrices (A and B).
    This reduces the number of parameters compared to a full-rank layer.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRank, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias

        # Initialize weight matrices A (in_features x rank) and B (rank x out_features)
        wA = torch.empty(self.in_features, rank)
        wB = torch.empty(self.rank, self.out_features)
        self.A = nn.Parameter(nn.init.kaiming_uniform_(wA))
        self.B = nn.Parameter(nn.init.kaiming_uniform_(wB))

        # Initialize bias if required
        if self.bias:
            wb = torch.empty(self.out_features)
            self.b = nn.Parameter(nn.init.uniform_(wb))

    def forward(self, x):
        # Apply low-rank transformation: X * A * B
        out = x @ self.A
        out = out @ self.B
        if self.bias:
            out += self.b  # Add bias if enabled
        return out

class Model(nn.Module):
    """
    Implements the HADL framework with optional Haar wavelet transformation, Discrete Cosine Transform (DCT) and Low Rank Approximation.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length
        self.pred_len = configs.pred_len  # Prediction horizon
        self.channels = configs.enc_in  # Number of input channels (features)
        self.rank = configs.rank  # Rank for low-rank approximation
        self.bias = configs.bias  # Whether to include bias
        self.individual = configs.individual  # Use separate models per channel
        self.enable_Haar = configs.enable_Haar  # Enable Haar transformation
        self.enable_DCT = configs.enable_DCT  # Enable Discrete Cosine Transform
        self.enable_iDCT = configs.enable_iDCT  # Enable Inverse Discrete Cosine Transform
        self.enable_lowrank = configs.enable_lowrank  # Enable low-rank approximation

        # Define low-pass filter for Haar decomposition (averaging adjacent values)
        self.low_pass_filter = torch.tensor([1, 1], dtype=torch.float32) / math.sqrt(2)
        self.low_pass_filter = self.low_pass_filter.reshape(1, 1, -1).repeat(self.channels, 1, 1)

        # Adjust input length if Haar decomposition is enabled
        if self.enable_Haar:
            in_len = (self.seq_len // 2) + 1 if (self.seq_len % 2) != 0 else (self.seq_len // 2)
        else:
            in_len = self.seq_len  # No Haar decomposition, use full sequence length

        # Initialize prediction layer(s)
        if self.individual:
            if self.enable_lowrank:
                self.pred_layer = nn.ModuleList([
                LowRank(in_features=in_len, 
                        out_features=self.pred_len, 
                        rank=self.rank, 
                        bias=self.bias)
                for _ in range(self.channels)
                ])
            else:
                self.pred_layer = nn.ModuleList([
                nn.Linear(in_features=in_len, 
                        out_features=self.pred_len, 
                        bias=self.bias)
                for _ in range(self.channels)
                ])
        else:
            if self.enable_lowrank:
                self.pred_layer = LowRank(in_features=in_len, 
                                        out_features=self.pred_len, 
                                        rank=self.rank, 
                                        bias=self.bias)
            else:
                self.pred_layer = nn.Linear(in_features=in_len, 
                                        out_features=self.pred_len, 
                                        bias=self.bias)

    def forward(self, x):
        """
        Forward pass of the model.
        x: Input tensor of shape [Batch, Input length, Channel]
        Returns: Output tensor of shape [Batch, Output length, Channel]
        """
        batch_size, _, _ = x.shape

        # Transpose input to [Batch, Channel, Input length]
        x = x.permute(0, 2, 1)
        
        # Compute mean for normalization
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = x - seq_mean  # Normalize input
        
        # Apply Haar transformation (low-pass filtering) if enabled
        if self.enable_Haar:
            if self.seq_len % 2 != 0:
                x = F.pad(x, (0, 1))  # Pad if sequence length is odd
            
            self.low_pass_filter = self.low_pass_filter.to(x.device)  # Move filter to same device as input
            x = F.conv1d(input=x, weight=self.low_pass_filter, stride=2, groups=self.channels)
        
        # Apply Discrete Cosine Transform (DCT) if enabled
        if self.enable_DCT:
            x = DiscreteCosineTransform.apply(x) / x.shape[-1]  # Scale DCT output
        
        # Prediction
        if self.individual:
            out = torch.empty(batch_size, self.channels, self.pred_len, device=x.device)
            for i in range(self.channels):
                pred = self.pred_layer[i](x[:, i, :].view(batch_size, 1, -1))
                out[:, i, :] = pred.view(batch_size, -1)
        else:
            out = self.pred_layer(x)

        # Apply Inverse DCT if enabled
        if self.enable_iDCT:
            out = iDiscreteCosineTransform.apply(out)  # Apply Inverse DCT

        # De-normalize the output by adding back the mean
        out = out + seq_mean

        return out.permute(0, 2, 1)  # Return output as [Batch, Output length, Channel]

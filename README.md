# HADL (Haar DCT Low-Rank) Framework

## Overview
HADL (Haar and Discrete Cosine Transform with Low-Rank Approximation) is a PyTorch-based framework that combines Haar wavelet transformation, Discrete Cosine Transform (DCT), and low-rank approximation for efficient and effective time series prediction for Long Term Time Series Forecasting.

## Features
- Haar Transform: Reduces noise by applying a low-pass filter.
- Discrete Cosine Transform (DCT): Captures periodic patterns in the frequency domain.
- Low-Rank Approximation (Optional): Efficiently models relationships using low-rank matrices.
- General layer for prediction.


## Components

### DiscreteCosineTransform
A custom autograd function for performing Discrete Cosine Transform (DCT) using scipy.

### LowRank
A low-rank approximation layer that reduces the number of parameters by using two smaller weight matrices.

### Model
The main model class that integrates Haar wavelet transformation, DCT, and low-rank approximation. It supports both individual and shared prediction layers for different channels.

## Usage

### Initialization
The model can be initialized with a configuration object that specifies various parameters such as sequence length, prediction length, number of channels, rank for low-rank approximation, and flags to enable/disable Haar and DCT transformations.

```python
configs = {
    'seq_len': 512,
    'pred_len': 96,
    'enc_in': 7,
    'rank': 30,
    'bias': True,
    'individual': False,
    'enable_Haar': True,
    'enable_DCT': True,
    'enable_lowrank': True
}

model = Model(configs)
```

### Forward Pass
The forward method takes an input tensor of shape `[Batch, Input length, Channel]` and returns an output tensor of shape `[Batch, Output length, Channel]`.

```python
input_tensor = torch.randn(512, 96, 10)  # Example input
output_tensor = model(input_tensor)
```

## License
This project is licensed under the MIT License.

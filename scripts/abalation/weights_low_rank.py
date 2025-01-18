import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import plotly.graph_objects as go

model_LR = torch.load("checkpoints/ETTh1_512_96_HaarDCT_ETTh1_ftM_sl512_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtlinear_True_Exp/checkpoint.pth")
model_S = torch.load("checkpoints/ETTh1_512_96_woLowRank_ETTh1_ftM_sl512_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtlinear_True_Exp/checkpoint.pth")

print(model_S.keys())
print(model_LR.keys())

A = model_LR["layer_lo.A"]
B = model_LR["layer_lo.B"]

W = model_S["layer_lo.weight"]


AB = A @ B
AB = AB.detach().numpy() 
AB = AB / np.max(np.abs(AB))

W = W.detach().numpy() 
W = W / np.max(np.abs(W))

W = W.T

print(np.max(AB), np.min(AB))
print(np.max(W), np.min(W))

fig = go.Figure(data=go.Heatmap(
    z=AB,
    colorscale='Viridis',
    zmin=np.min(AB),
    zmax=np.max(AB)
      # Change colormap (e.g., 'Plasma', 'Cividis', 'RdBu')
))

# Set x and y axis ticks
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, AB.shape[1], 10),  # Tick positions at multiples of 24
        ticktext=np.arange(0, AB.shape[1], 10).astype(str),  # Convert to string for labels
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, AB.shape[0], 20),  # Tick positions at multiples of 24
        ticktext=np.arange(0, AB.shape[0], 20).astype(str),  # Convert to string for labels
        tickfont=dict(size=14)
    )
)

# Update title and layout
fig.update_layout(

)

# Save the plot as an image (ensure kaleido is installed for image export)
fig.write_image("AB_matrix.png", width=800, height=800, scale=2)


fig = go.Figure(data=go.Heatmap(
    z=W,
    colorscale='Viridis',
    zmin=np.min(W),
    zmax=np.max(W)
      # Change colormap (e.g., 'Plasma', 'Cividis', 'RdBu')
))

# Set x and y axis ticks
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, W.shape[1], 10),  # Tick positions at multiples of 24
        ticktext=np.arange(0, W.shape[1], 10).astype(str),  # Convert to string for labels,
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, W.shape[0], 20),  # Tick positions at multiples of 24
        ticktext=np.arange(0, W.shape[0], 20).astype(str),  # Convert to string for labels
        tickfont=dict(size=14)
    )
)

# Update title and layout
fig.update_layout(

)

# Save the plot as an image (ensure kaleido is installed for image export)
fig.write_image("W_matrix.png", width=800, height=800, scale=2)



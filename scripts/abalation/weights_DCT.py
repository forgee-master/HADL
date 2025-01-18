import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import plotly.graph_objects as go

model_w = torch.load("checkpoints/ETTh1_512_96_HaarDCT_ETTh1_ftM_sl512_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtlinear_True_Exp/checkpoint.pth")
model_wo = torch.load("checkpoints/ETTh1_512_96_woDCT_ETTh1_ftM_sl512_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtlinear_True_Exp/checkpoint.pth")

A1 = model_w["layer_lo.A"]
B1 = model_w["layer_lo.B"]

A2 = model_wo["layer_lo.A"]
B2 = model_wo["layer_lo.B"]



ABw = A1 @ B1
ABw = ABw.detach().numpy() 
ABw = ABw / np.max(np.abs(ABw))


ABwo = A2 @ B2
ABwo = ABwo.detach().numpy() 
ABwo = ABwo / np.max(np.abs(ABwo))

print(np.max(ABw), np.min(ABw))
print(np.max(ABwo), np.min(ABwo))

fig = go.Figure(data=go.Heatmap(
    z=ABw,
    colorscale='Viridis',
    zmin=np.min(ABw),
    zmax=np.max(ABw)
      # Change colormap (e.g., 'Plasma', 'Cividis', 'RdBu')
))

# Set x and y axis ticks
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, ABw.shape[1], 10),  # Tick positions at multiples of 24
        ticktext=np.arange(0, ABw.shape[1], 10).astype(str),  # Convert to string for labels
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, ABw.shape[0], 20),  # Tick positions at multiples of 24
        ticktext=np.arange(0, ABw.shape[0], 20).astype(str),  # Convert to string for labels
        tickfont=dict(size=14)
    )
)

# Update title and layout
fig.update_layout(

)

# Save the plot as an image (ensure kaleido is installed for image export)
fig.write_image("ABw_matrix.png", width=800, height=800, scale=2)


fig = go.Figure(data=go.Heatmap(
    z=ABwo,
    colorscale='Viridis',
    zmin=np.min(ABwo),
    zmax=np.max(ABwo)
      # Change colormap (e.g., 'Plasma', 'Cividis', 'RdBu')
))

# Set x and y axis ticks
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, ABwo.shape[1], 10),  # Tick positions at multiples of 24
        ticktext=np.arange(0, ABwo.shape[1], 10).astype(str),  # Convert to string for labels,
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=np.arange(0, ABwo.shape[0], 20),  # Tick positions at multiples of 24
        ticktext=np.arange(0, ABwo.shape[0], 20).astype(str),  # Convert to string for labels
        tickfont=dict(size=14)
    )
)

# Update title and layout
fig.update_layout(

)

# Save the plot as an image (ensure kaleido is installed for image export)
fig.write_image("ABwo_matrix.png", width=800, height=800, scale=2)



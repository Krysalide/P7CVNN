
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from Experimental.learnable_fft_wip2 import SignalProcessLayer

st.set_page_config(layout="wide")
st.title("🔍 View Weights from SignalProcessLayer")

model = SignalProcessLayer(use_fft_weights=True)
model.eval()


doppler_weights = model.get_doppler_weights().detach().cpu().numpy()
range_weights = model.get_range_weights().detach().cpu().numpy()

range_hamming=model.get_window_range_coeff().detach().cpu().numpy()
doppler_hamming=model.get_window_doppler_coeff().detach().cpu().numpy()

#view_option = st.radio("Select weights to view:", ("Doppler Weights", "Range Weights","Dopller Window","Range Window"))
view_option = st.radio(
    "Select weights to view:",
    ("Select a view", "Doppler Weights", "Range Weights", "Dopller Window", "Range Window"),
    index=0 # This makes "Select a view" the default selected option
)

def render_heatmap(data, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    heatmap = ax.imshow(data, aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    fig.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8)
    st.pyplot(fig)

def render_surface(data, title):

    rows, cols = data.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    Z = data

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Columns",
            yaxis_title="Rows",
            zaxis_title="Values"
        ),
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# Display only the selected heatmap
if view_option == "Doppler Weights":
    render_heatmap(doppler_weights, "Doppler Weights")
elif view_option=='Range Weights':
    render_heatmap(range_weights, "Range Weights")
elif view_option=='Dopller Window':
    render_surface(doppler_hamming.squeeze(axis=-1),title='Doppler Hamming 3D view')
elif view_option=='Range Window':
    render_surface(range_hamming.squeeze(axis=-1),title='Range Hamming 3D view')
elif  view_option=='Select a view':
    st.info("Please select an option from the radio buttons above to view the weights.")









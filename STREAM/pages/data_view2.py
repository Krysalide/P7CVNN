import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

# Assuming these imports are correct and available in your environment
from Experimental.data_fft_reader import RadarFFTDataset

st.set_page_config(layout="wide")
st.title('DATASET EXPLORER')

# --- Data Loading and Preprocessing ---
data_folder = '/home/christophe/RADIalP7/SMALL_DATASET/TEST'
assert os.path.exists(data_folder), 'Data folder not found. Please check the path.'

element_number = 60 # Maximum number of elements to load
assert element_number <= 60, f"Number of elements is limited to 60. You requested {element_number}."
indices = list(range(element_number)) # Specify number of elements

dataset = RadarFFTDataset(data_folder, indices)
st.sidebar.write(f"Dataset loaded with {len(dataset)} samples.")

# --- Helper Function for Heatmap Plotting ---
def render_heatmap(data, title):
    """Renders a heatmap using Matplotlib and displays it in Streamlit."""
    fig, ax = plt.subplots(figsize=(4, 3))
    heatmap = ax.imshow(data, aspect='auto', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("Dimension 2 (Columns)")
    ax.set_ylabel("Dimension 1 (Rows)")
    fig.colorbar(heatmap, ax=ax, orientation="vertical", shrink=0.8)
    st.pyplot(fig)
    plt.close(fig) # Always close figures to prevent memory leaks

# --- User Selection for Sample Index (Primary Control) ---
st.sidebar.header("Sample Selection")
# Let the user choose the sample index, starting from 0 (no selection)
selected_sample_index = st.sidebar.slider(
    "Select Sample Index:",
    min_value=0, # Use 0 as an 'unselected' or 'initial' state
    max_value=len(dataset) - 1, # Max index of available samples
    value=0, # Default value will be the first index (or 0 for 'no selection' if min_value is used this way)
    step=1
)

# --- Conditional Rendering based on Sample Index ---
if selected_sample_index == 0: # Or if you want a true "no selection" state, use None and check for None
    st.info("Please select a sample index from the slider on the left to view data.")
else:
    # If a sample is selected, display its index
    st.write(f"Displaying data for selected sample index: **{selected_sample_index}**")

    # --- Load data for the selected sample ---
    raw_adc_test = dataset.get_adc(idx=selected_sample_index)
    range_doppler_test = dataset.get_range_doppler(idx=selected_sample_index)

    # --- Antenna Selection (Secondary Control, appears after sample is chosen) ---
    st.sidebar.subheader("Antenna Selection")
    antenna = st.sidebar.slider(
        "Select Antenna Number:",
        min_value=1,
        max_value=raw_adc_test.shape[-1], # Dynamically set max based on loaded data
        value=1, # Default to the first antenna
        step=1
    )
    selected_antenna_idx = antenna - 1 # Convert to 0-based index for NumPy

    # --- Data Slicing and Plotting ---
    if raw_adc_test.shape[-1] <= selected_antenna_idx or range_doppler_test.shape[-1] <= selected_antenna_idx:
        st.error(f"Error: Selected antenna {antenna} is out of bounds for the loaded data. "
                 f"Data has {raw_adc_test.shape[-1]} antennas (0 to {raw_adc_test.shape[-1]-1}).")
    else:
        # Slice data for the selected antenna
        antenna_view = raw_adc_test[:, :, selected_antenna_idx]
        doppler_view = range_doppler_test[:, :, selected_antenna_idx]

        # Display Heatmaps
        st.subheader(f"Heatmap for Raw ADC Data Magnitude (Antenna {antenna})")
        render_heatmap(np.abs(antenna_view), f"Raw ADC Magnitude (Antenna {antenna})")

        st.subheader(f"Heatmap for Range-Doppler Magnitude Data (Antenna {antenna})")
        render_heatmap(np.abs(doppler_view), f"Range-Doppler Magnitude (Antenna {antenna})")

        st.subheader(f"Heatmap for Raw ADC Data Phase (Antenna {antenna})")
        render_heatmap(np.angle(antenna_view), f"Raw ADC Phase (Antenna {antenna})")

        st.subheader(f"Heatmap for Range-Doppler Phase Data (Antenna {antenna})")
        render_heatmap(np.angle(doppler_view), f"Range-Doppler Phase (Antenna {antenna})")
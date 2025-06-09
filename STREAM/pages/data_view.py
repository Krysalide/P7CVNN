import streamlit as st
import os
import numpy as np # Import numpy for array operations
import matplotlib.pyplot as plt # Import matplotlib for heatmap plotting

# Assuming these imports are correct and available in your environment
from Experimental.data_fft_reader import RadarFFTDataset


st.set_page_config(layout="wide")
st.title('DATASET PAGE')

# --- Data Loading and Preprocessing ---
data_folder = '/home/christophe/RADIalP7/SMALL_DATASET/TEST'
assert os.path.exists(data_folder), 'Data folder not found. Please check the path.'

element_number = 60
assert element_number <= 60, f"Number of elements is limited to 60. You requested {element_number}."
indices = list(range(element_number)) # specify number of elements

dataset = RadarFFTDataset(data_folder, indices)
st.write(f"Dataset loaded with {len(dataset)} samples (showing first {element_number}).")

# --- Fixed Test Index for Demonstration ---
# You might want to make this user-selectable later as well!
test_indice = 5
st.write(f"Displaying data for test sample index: {test_indice}")

raw_adc_test = dataset.get_adc(idx=test_indice)
range_doppler_test = dataset.get_range_doppler(idx=test_indice)

# --- Streamlit UI for Antenna Selection ---
st.subheader("Antenna Selection")


antenna = st.slider(
    "Select Antenna Number:",
    min_value=1,
    max_value=16,
    value=5, # Default value
    step=1
)


selected_antenna_idx = antenna - 1


if raw_adc_test.shape[-1] <= selected_antenna_idx or range_doppler_test.shape[-1] <= selected_antenna_idx:
    st.error(f"Error: Selected antenna {antenna} is out of bounds for the loaded data. "
             f"Data has {raw_adc_test.shape[-1]} antennas (0 to {raw_adc_test.shape[-1]-1}).")
else:
    
    antenna_view = raw_adc_test[:, :, selected_antenna_idx]
    
    doppler_view = range_doppler_test[:, :, selected_antenna_idx]
    
    # --- Heatmap Plotting Function ---
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

    # --- Display Heatmaps ---
    st.subheader(f"Heatmap for Raw ADC Data magnitude (Antenna {antenna})")
    render_heatmap(np.abs(antenna_view), f"Raw ADC View (Antenna {antenna})")

    st.subheader(f"Heatmap for Range-Doppler magnitude Data (Antenna {antenna})")
    render_heatmap(np.abs(doppler_view), f"Range-Doppler View (Antenna {antenna})")

    st.subheader(f"Heatmap for Raw ADC Data phase (Antenna {antenna})")
    render_heatmap(np.angle(antenna_view), f"Raw ADC View (Antenna {antenna})")

    st.subheader(f"Heatmap for Range-Doppler phase Data (Antenna {antenna})")
    render_heatmap(np.angle(doppler_view), f"Range-Doppler View (Antenna {antenna})")
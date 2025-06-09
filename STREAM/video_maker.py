import os

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import shutil

def get_sorted_npy_paths(folder_path):
    # Match pattern like: second_fft_91.npy
    pattern = re.compile(r"second_fft_(\d+)\.npy")
    npy_files = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            full_path = os.path.join(folder_path, filename)
            npy_files.append((number, full_path))
    
    # Sort by the number and return the paths
    npy_files.sort(key=lambda x: x[0])
    return [path for _, path in npy_files]

range_doppler_file_folder=f'/home/christophe/RADIalP7/SMALL_DATASET/VIDEO/FFT2'

assert os.path.exists(range_doppler_file_folder)

output_video_path = 'range_doppler_heatmap_animation.mp4'
temp_frames_folder = 'temp_heatmap_frames'

# Create a temporary folder for saving frames if it doesn't exist
os.makedirs(temp_frames_folder, exist_ok=True)

assert os.path.exists(range_doppler_file_folder), f"Folder not found: {range_doppler_file_folder}"

rd_files = get_sorted_npy_paths(range_doppler_file_folder)
antenna_id = 7 # The antenna ID you want to visualize

# Determine min/max values for consistent colormap across all frames
# This helps with visual consistency in the video
all_range_doppler_data = []
for rd_file in rd_files:
    range_doppler = np.load(rd_file)
    sample_rad_doppler = np.abs(range_doppler[:, :, antenna_id])
    
    all_range_doppler_data.append(sample_rad_doppler)

# Convert to a single numpy array to easily find global min/max
all_range_doppler_data = np.array(all_range_doppler_data)
global_min_val = np.min(all_range_doppler_data)
global_max_val = np.max(all_range_doppler_data)

print(f"Generating {len(rd_files)} heatmap frames...")

for i, rd_file in enumerate(rd_files):
    range_doppler = np.load(rd_file)
    sample_rad_doppler = np.abs(range_doppler[:, :, antenna_id])

    fig, ax = plt.subplots(figsize=(8, 6)) # Adjust figure size as needed

    # Create the heatmap
    # Using imshow for 2D array heatmaps
    # vmin and vmax are crucial for consistent color scaling across frames
    cax = ax.imshow(sample_rad_doppler, cmap='viridis', aspect='auto',
                    vmin=global_min_val, vmax=global_max_val)

    ax.set_title(f'Range-Doppler Heatmap - Frame {i+1}')
    ax.set_xlabel('Doppler Bins')
    ax.set_ylabel('Range Bins')

    # Add a colorbar
    if i == 0: # Only add colorbar once if you want it to be static, or for each frame
        fig.colorbar(cax, label='Intensity')

    # Save the plot as an image
    frame_filename = os.path.join(temp_frames_folder, f'heatmap_frame_{i:04d}.png')
    plt.savefig(frame_filename, bbox_inches='tight', dpi=100) # dpi can be adjusted for quality
    plt.close(fig) # Close the figure to free up memory

print("Heatmap frames generated.")

frame_files = [os.path.join(temp_frames_folder, f) for f in os.listdir(temp_frames_folder) if f.startswith('heatmap_frame_') and f.endswith('.png')]
frame_files.sort() # Ensure correct order

if not frame_files:
    print("No frames found to create video. Exiting.")
else:
    print(f"Creating video from {len(frame_files)} frames using imageio...")
    with imageio.get_writer(output_video_path, mode='I', fps=10) as writer: # Adjust fps as needed
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Video saved to {output_video_path}")

    # Clean up temporary frames (optional)
    
    shutil.rmtree(temp_frames_folder)
    print(f"Temporary frames folder '{temp_frames_folder}' removed.")









# rd_files=get_sorted_npy_paths(range_doppler_file_folder)
# antenna_id=7
# for rd_file in rd_files:
#     range_doppler=np.load(rd_file)
#     sample_rad_doppler=range_doppler[:,:,antenna_id]

    # print(type(sample_rad_doppler))
    # print(sample_rad_doppler.shape)
    # sys.exit()


# def create_heatmap_video(npy_paths, antenna_id, output_video_path, fps=10):
#     images = []
    
#     for path in npy_paths:
#         rd_data = np.load(path)
#         heatmap_data = rd_data[:, :, antenna_id]
        
#         # Plot the heatmap
#         fig, ax = plt.subplots()
#         im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis')
#         plt.axis('off')  # Hide axes for clean image

#         # Save plot to buffer
#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         images.append(image)
#         plt.close(fig)

#     # Define video properties
#     height, width, _ = images[0].shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

#     for img in images:
#         out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#     out.release()
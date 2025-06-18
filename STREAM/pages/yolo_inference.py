import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import time

# Title
st.title('YOLOV10 INFERENCES ON RADIAL DATASET SEQUENCE')
st.write('---')
st.write("please note we have been obliged to add delay between sequences")
st.write("Yolo model is able to do predictions at rates faster than reality")
st.write('In inference mode yolov10 uses 2GB of Gpu memory')
st.write('---')

# Load the model
model = YOLO("yolov10n.pt")


full_coco_class_map = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'airplane': 4,
    'bus': 5,
    'train': 6,
    'truck': 7,
    'traffic light': 9,
    'fire hydrant': 10,
    'stop sign': 11,
    'parking meter': 12,
    'cat': 15,
    'dog': 16,
}

# Sidebar multiselect with only relevant class names
selected_class_names = st.sidebar.multiselect(
    "Select classes to detect:",
    options=list(full_coco_class_map.keys()),
    default=["person", "car","bicycle","motorcycle","bus","truck","traffic light"]
)

# Map selected names to original COCO indices
selected_class_indices = [full_coco_class_map[name] for name in selected_class_names]

# coco_classes = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'traffic light', 'fire hydrant',
#     'stop sign', 'parking meter','cat', 'dog'
# ]

# selected_class_names = st.sidebar.multiselect(
#     "Select classes to detect:",
#     options=coco_classes,
#     default=["person", "car"]  # optional default
# )

# selected_class_indices = [coco_classes.index(name) for name in selected_class_names]

st.write("---")

# video we want to pass to yolov10

video_options = {
    "Video 1": "/home/christophe/RADIalP7/DATASET/RECORD@2020-11-22_12.20.19/RECORD@2020-11-22_12.20.19_camera.mjpg",
    "Video 2": "/home/christophe/RADIalP7/DATASET/RECORD@2020-11-21_13.57.07/RECORD@2020-11-21_13.57.07_camera.mjpg",
    "Video 3": "/home/christophe/RADIalP7/DATASET/RECORD@2020-11-21_11.54.31/RECORD@2020-11-21_11.54.31_camera.mjpg"
}
#video_path = "/home/christophe/ComplexNet/STREAM/intro.mp4"

#video_path="/home/christophe/RADIalP7/DATASET/RECORD@2020-11-21_13.57.07/RECORD@2020-11-21_13.57.07_camera.mjpg"

#video_path="/home/christophe/RADIalP7/DATASET/RECORD@2020-11-21_11.54.31/RECORD@2020-11-21_11.54.31_camera.mjpg"

selected_video_label = st.sidebar.selectbox("Choose a video", list(video_options.keys()))
selected_video_path = video_options[selected_video_label]

cap = cv2.VideoCapture(selected_video_path)

stframe = st.empty()
frame_skip = 1  

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(0.025)
    frame_idx += 1
    if frame_idx % frame_skip != 0:
        continue

    # Convert BGR to RGB for YOLO 
    # TODO test if necessary? 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO prediction
    results = model.predict(source=frame_rgb, save=False, conf=0.3, verbose=False,
                            classes=selected_class_indices)

    # Plot boxes on the frame
    annotated_frame = results[0].plot()  # Returns annotated image (numpy)

    # Show frame in Streamlit
    stframe.image(annotated_frame, channels="RGB", use_container_width=True)

cap.release()

st.write("---")



# import streamlit as st
# import streamlit.components.v1 as components
# import base64
# from ultralytics import YOLO

# st.title('RADAR APPLICATION')

# st.write("---")

# model = YOLO("yolov10n.pt")
# video_path = "/home/christophe/ComplexNet/STREAM/intro.mp4"
# video_file = open(video_path, 'rb')
# video_bytes = video_file.read()
# video_base64 = base64.b64encode(video_bytes).decode()

# components.html(f"""
# <video autoplay loop muted playsinline width="640" height="480">
#   <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
#   Your browser does not support the video tag.
# </video>
# """, height=500)

# st.write("---")




# gif_path = "output_range_dopller.gif"

# try:
#     with open(gif_path, "rb") as file:
#         gif_bytes = file.read()
    
#     # Encode the GIF to base64
#     gif_base64 = base64.b64encode(gif_bytes).decode()
    
#     # Use st.markdown to insert an HTML <img> tag with the base64 GIF
#     st.header("Animated GIF")
#     st.markdown(
#         f'<img src="data:image/gif;base64,{gif_base64}" alt="Animated GIF" style="width:100%;">',
#         unsafe_allow_html=True,
#     )
#     st.caption("My animated GIF (using HTML markdown)")

# except FileNotFoundError:
#     st.warning(f"Please place the GIF file '{gif_path}' in the same directory as your script.")

# st.write("---")
# st.write("You can adjust the `caption` and styling as needed.")


# try:
#     with open("output_range_dopller.gif", "rb") as file:
#         local_gif = file.read()
#     st.header("Range Doppler maps")
#     st.image(local_gif, caption="Animated view of phase of antenna 7 range doppler", use_container_width=True,output_format='auto')
# except FileNotFoundError:
#     st.warning("Please place a GIF file named 'my_gif.gif' in the same directory as your script for the local GIF example to work.")

# st.write("---")
# st.image(image='/home/christophe/ComplexNet/STREAM/auto_radar.jpg',width=400)
# st.write("---")









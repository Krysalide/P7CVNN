
import streamlit as st
import streamlit.components.v1 as components
import base64

st.title('RADAR APPLICATION')

video_path = "/home/christophe/ComplexNet/STREAM/intro.mp4"
video_file = open(video_path, 'rb')
video_bytes = video_file.read()
video_base64 = base64.b64encode(video_bytes).decode()

components.html(f"""
<video autoplay loop muted playsinline width="640" height="480">
  <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
  Your browser does not support the video tag.
</video>
""", height=500)










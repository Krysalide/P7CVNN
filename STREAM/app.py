
import streamlit as st
import streamlit.components.v1 as components
import base64


st.title('RADAR APPLICATION')

st.write("---")

st.write("---")
st.image(image='/home/christophe/ComplexNet/STREAM/auto_radar.jpg',width=400)
st.write("---")


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

st.write("---")


try:
    with open("output_range_dopller.gif", "rb") as file:
        local_gif = file.read()
    st.header("Range Doppler maps")
    st.image(local_gif, caption="Animated view of phase of antenna 7 range doppler", use_container_width=True,output_format='auto')
except FileNotFoundError:
    st.warning("Please place a GIF file named 'my_gif.gif' in the same directory as your script for the local GIF example to work.")











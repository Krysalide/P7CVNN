
import streamlit as st
import streamlit.components.v1 as components
import base64

st.image(image='/home/christophe/ComplexNet/STREAM/auto_radar.jpg',width=400)
st.write("---")
st.title('RADAR APPLICATION')

# file = open("output_range_dopller.gif", 'rb')
# contents = file.read()
# data_url = base64.b64encode(contents).decode('utf-8-sig')
# file.close()
# st.markdown(f'<img src="data:image/gif;base64,{data_url}>',unsafe_allow_html = True)
st.write("---")

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


try:
    with open("output_range_dopller.gif", "rb") as file:
        local_gif = file.read()
    st.header("Range Doppler maps")
    st.image(local_gif, caption="Animated view of phase of antenna 7 range doppler", use_container_width=True,output_format='auto')
except FileNotFoundError:
    st.warning("Please place a GIF file named 'my_gif.gif' in the same directory as your script for the local GIF example to work.")

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









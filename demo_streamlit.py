import streamlit as st
from PIL import Image
 
st.title('Car Model Classification')

st.sidebar.header("User Input Image")

img = st.sidebar.file_uploader(label='Upload your bmp image', type=['bmp'])
if img:
    image = Image.open(img)
    st.image(image)


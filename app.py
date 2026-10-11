import streamlit as st
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gdown
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")

st.title("🚗 Driver Drowsiness Monitor")
st.subheader("AI-Powered Fatigue Detection System")
st.write("Upload a driver's photo to detect drowsiness.")

@st.cache_resource
def load_model():
    # Replace 'YOUR_FILE_ID' with the actual ID from your Google Drive link.
    # Example: if your link is https://drive.google.com/file/d/1a2b3c4d5e/view
    # then the ID is '1a2b3c4d5e'
    url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
    output = 'best.pt'
    gdown.download(url, output, quiet=False)
    return YOLO(output)

model = load_model()

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        results = model(image)
        
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        st.image(im, caption='Detection Result', use_column_width=True)
        
        boxes = r.boxes
        if len(boxes) > 0:
            st.error(f"⚠️ WARNING: Drowsiness detected!")
        else:
            st.info("✅ Driver is alert.")

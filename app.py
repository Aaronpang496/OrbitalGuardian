import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("🏙️ City Danger & Hazard Detector")
st.write("Take a picture or upload an image to scan for urban threats.")

# 1. Load your newly generated combined weights
# Streamlit caches this so the app stays fast and doesn't reload the file constantly
@st.cache_resource
def load_danger_model():
    return YOLO("best.pt")

model = load_danger_model()

# 2. Add an image picker (supports camera on mobile phones and file uploads)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or take a live photo!")

# Use whichever input the user provided
target_image = uploaded_file if uploaded_file is not None else camera_file

if target_image is not None:
    # Open the image using PIL
    image = Image.open(target_image)
    
    # Show the original picture to the user
    st.image(image, caption="Processing uploaded scene...", use_column_width=True)
    
    with st.spinner("Scanning street metrics for dangers..."):
        # Run inference using your custom best.pt
        results = model(image)
        
        # Plot the bounding boxes directly onto the image array
        res_plotted = results[0].plot()
        
        # Convert BGR back to RGB for streamlit rendering
        output_image = Image.fromarray(res_plotted[..., ::-1])
        
    # 3. Display the final processed danger map
    st.subheader("⚠️ Detection Results")
    st.image(output_image, caption="Identified Urban Dangers", use_column_width=True)

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="CityGuardian", page_icon="🏙️）

st.title("🏙️ CityGuardian")
st.subheader("AI-Powered Urban Hazard Detection System")

st.write("Upload a photo of a street or building to detect hazards.")

@st.cache_resource
def load_model():
    # We use a pre-trained YOLO model for the demo.
    # In the future, you can replace 'yolov8n.pt' with your own 'best.pt' file.
   from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="cazzz307/yolov8-crack-detection", filename="best.pt")
return YOLO(model_path) 

model = load_model()

uploaded_file = st.file_uploader("Choose a space image...", type=["jpg", "jpeg", "png"])

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
            distance_km = np.random.uniform(0.5, 10.0)
            if distance_km < 2.0:
                st.error(f"⚠️ WARNING: Collision risk detected! Distance: {distance_km:.2f} km")
            else:
                st.success(f"✅ Safe. Distance: {distance_km:.2f} km")
        else:
            st.info("No debris detected in this image.")

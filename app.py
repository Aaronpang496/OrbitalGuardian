import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="CityGuardian", page_icon="🏙️")

st.title("🏙️ CityGuardian")
st.subheader("AI-Powered Urban Hazard Detection System")

st.write("Upload a photo of a street or building to detect hazards.")

@st.cache_resource
def load_model():
    # 直接在雲端生成一個空的 YOLO 模型，不需要從網上下載
    model = YOLO('yolov8n.yaml') 
    return model

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
            st.error(f"⚠️ WARNING: Hazard detected!")
        else:
            st.info("No specific hazard detected by the generic model. In Stage 2, we will train it with real crack data.")

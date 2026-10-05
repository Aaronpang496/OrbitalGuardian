import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="CityGuardian", page_icon="🏙️")

st.title("🏙️ CityGuardian")
st.subheader("AI-Powered Urban Hazard Detection System")
st.write("Upload a photo of a street or building to detect hazards.")

@st.cache_resource
def load_model():
    try:
        # 嘗試下載公開的裂縫偵測模型
        model_path = hf_hub_download(
            repo_id="keremberke/yolov8m-crack-detection",
            filename="best.pt"
        )
        return YOLO(model_path)
    except Exception as e:
        # 如果下載失敗，使用內建備用模型，保證不會崩潰
        st.warning("Could not download the crack model. Using a generic model instead.")
        return YOLO('yolov8n.pt')

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
            st.error(f"⚠️ WARNING: {len(boxes)} hazard(s) detected!")
        else:
            st.info("No hazard detected in this image.")

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
    # 我們列出三個可能的公開模型，從第一個開始試，哪個成功就用哪個
    candidate_models = [
        ("cazzz307/yolov8-crack-detection", "best.pt"),
        ("keremberke/yolov8m-crack-detection", "best.pt"),
        ("hf-vision/crack-detection", "best.pt")
    ]
    
    for repo_id, filename in candidate_models:
        try:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            return YOLO(model_path)
        except Exception:
            continue # 如果這個失敗，就試下一個
    
    # 如果全部失敗，就用內建的通用模型作為最後防線
    st.warning("Could not load a specialized model. Using a generic fallback model.")
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
            st.error(f"⚠️ WARNING: {len(boxes)} potential hazard(s) detected!")
        else:
            st.info("No specific hazard detected. In Stage 2, we will train our own model with local data.")

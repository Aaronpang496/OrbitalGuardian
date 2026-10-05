import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import os

st.set_page_config(page_title="CityGuardian", page_icon="🏙️")
st.title("🏙️ CityGuardian")
st.subheader("AI-Powered Urban Hazard Detection System")
st.write("Upload a photo of a street or building to detect hazards.")

@st.cache_resource
def train_model():
    # 1. 生成 50 張「裂縫」訓練圖
    base_dir = 'crack_dataset'
    os.makedirs(f'{base_dir}/images/train', exist_ok=True)
    os.makedirs(f'{base_dir}/labels/train', exist_ok=True)
    
    for i in range(50):
        img = Image.new('RGB', (640, 640), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        x1, y1 = np.random.randint(100, 500, 2)
        x2, y2 = x1 + np.random.randint(100, 200), y1 + np.random.randint(100, 200)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=8)
        img.save(f'{base_dir}/images/train/img_{i}.jpg')
        with open(f'{base_dir}/labels/train/img_{i}.txt', 'w') as f:
            f.write(f'0 0.5 0.5 0.3 0.3')
            
    with open(f'{base_dir}/data.yaml', 'w') as f:
        f.write(f"path: /content/{base_dir}\ntrain: images/train\nval: images/train\nnames:\n  0: crack")

    # 2. 訓練模型
    model = YOLO('yolov8n.yaml')
    model.train(data=f'{base_dir}/data.yaml', epochs=10, imgsz=640, batch=4, name='crack_model', verbose=False)
    return YOLO(f'runs/detect/crack_model/weights/best.pt')

model = train_model()

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
            st.error(f"⚠️ WARNING: Crack detected!")
        else:
            st.info("No crack detected in this image.")

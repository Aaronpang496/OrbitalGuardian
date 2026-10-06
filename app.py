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
    base_dir = 'city_hazard_dataset'
    os.makedirs(f'{base_dir}/images/train', exist_ok=True)
    os.makedirs(f'{base_dir}/labels/train', exist_ok=True)
    
    hazards = ['crack', 'blocked_exit', 'fallen_tree', 'broken_sign', 'pothole']
    
    for i in range(200):
        img = Image.new('RGB', (640, 640), color=(180, 180, 180))
        draw = ImageDraw.Draw(img)
        hazard = np.random.choice(hazards)
        
        if hazard == 'crack':
            x1, y1 = np.random.randint(100, 500, 2)
            x2, y2 = x1 + np.random.randint(50, 150), y1 + np.random.randint(50, 150)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=6)
            label = 0
        elif hazard == 'blocked_exit':
            x1, y1 = np.random.randint(100, 400, 2)
            draw.rectangle([x1, y1, x1+120, y1+120], fill=(139, 69, 19))
            label = 1
        elif hazard == 'fallen_tree':
            x1, y1 = np.random.randint(100, 400, 2)
            draw.line([(x1, y1), (x1+200, y1+50)], fill=(34, 139, 34), width=15)
            label = 2
        elif hazard == 'broken_sign':
            x1, y1 = np.random.randint(100, 400, 2)
            draw.rectangle([x1, y1, x1+80, y1+80], fill=(255, 0, 0))
            draw.line([(x1, y1+80), (x1+40, y1+150)], fill=(0, 0, 0), width=5)
            label = 3
        else:
            x1, y1 = np.random.randint(100, 400, 2)
            draw.ellipse([x1, y1, x1+100, y1+80], fill=(50, 50, 50))
            label = 4
        
        img.save(f'{base_dir}/images/train/img_{i}.jpg')
        with open(f'{base_dir}/labels/train/img_{i}.txt', 'w') as f:
            f.write(f'{label} 0.5 0.5 0.3 0.3')
    
    with open(f'{base_dir}/data.yaml', 'w') as f:
        f.write(f"path: {os.path.abspath(base_dir)}\ntrain: images/train\nval: images/train\nnames:\n  0: crack\n  1: blocked_exit\n  2: fallen_tree\n  3: broken_sign\n  4: pothole")

    model = YOLO('yolov8n.yaml')
    model.train(data=f'{base_dir}/data.yaml', epochs=10, imgsz=640, batch=8, project='.', name='my_model', verbose=False)
    
    return YOLO('my_model/weights/best.pt')

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
            st.error(f"⚠️ WARNING: {len(boxes)} hazard(s) detected!")
        else:
            st.info("No hazard detected in this image.")

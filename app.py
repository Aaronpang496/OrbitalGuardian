import os
import tempfile
from pathlib import Path
import urllib.request

import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import cv2
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Orbital Guardian", page_icon="🛡️", layout="centered")
st.markdown("""
    <style>
    .main-title { font-size: 40px; font-weight: 800; color: #1e293b; margin-bottom: 0px; }
    .sub-title { font-size: 18px; color: #64748b; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🛡️ Orbital Guardian</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Driver Monitoring + Biometric Eye Tracker</p>', unsafe_allow_html=True)

# --- INITIALIZE AI MODELS ---
WORKSPACE = "tsz-chung-pang"
FACE_WORKFLOW = "drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-1-rfdetr-small-t1-logic"
POSTURE_WORKFLOW = "driver-head-on-wheel-posture"
FACE_THRESHOLD = 0.3

@st.cache_resource
def load_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="3zlKh7bl9yZXMl5RvxNY" 
    )

# Load OpenCV Face and Eye detectors safely (DOWNLOADING THE FILES)
@st.cache_resource
def load_cv_cascades():
    face_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    eye_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
    
    urllib.request.urlretrieve(face_url, "face.xml")
    urllib.request.urlretrieve(eye_url, "eye.xml")
    
    face_cascade = cv2.CascadeClassifier("face.xml")
    eye_cascade = cv2.CascadeClassifier("eye.xml")
    
    return face_cascade, eye_cascade

client = load_client()
face_cascade, eye_cascade = load_cv_cascades()

# --- HELPER FUNCTIONS ---
def get_boxes(result):
    if not isinstance(result, list) or not result: return []
    first = result[0]
    if not isinstance(first, dict): return []
    output = first.get("predictions", {})
    if isinstance(output, dict): boxes = output.get("predictions", [])
    elif isinstance(output, list): boxes = output
    else: boxes = []
    return [box for box in boxes if isinstance(box, dict)]

def has_class(boxes, class_name, minimum_confidence=0):
    return any(box.get("class") == class_name and isinstance(box.get("confidence"), (int, float)) and box["confidence"] >= minimum_confidence for box in boxes)

def draw_boxes(image, boxes, color):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        try:
            x, y, width, height = float(box["x"]), float(box["y"]), float(box["width"]), float(box["height"])
        except (KeyError, TypeError, ValueError): continue
        x1, y1, x2, y2 = x - width / 2, y - height / 2, x + width / 2, y + height / 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((max(0, x1), max(0, y1 - 18)), str(box.get("class", "detection")), fill=color)

# --- OPENCV BIOMETRIC TRACKER ---
def analyze_eyes_opencv(image):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, False, np.array(image)

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    img_annotated = img_cv.copy()
    cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 255), 2) 
    
    eye_count = 0
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img_annotated, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2) 
        eye_count += 1
        
    img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
    
    # If OpenCV finds 0 eyes, they are closed
    is_microsleep = (eye_count == 0)
    
    return eye_count, is_microsleep, img_annotated_rgb

# --- UI AND IMAGE INPUT ---
st.write("### Live Analysis")
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Live Camera"])
image_source = None

with tab1:
    uploaded_file = st.file_uploader("Upload a driver photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file).convert("RGB")

with tab2:
    camera_file = st.camera_input("Take a live photo", label_visibility="collapsed")
    if camera_file is not None:
        image_source = Image.open(camera_file).convert("RGB")

# --- PROCESSING ---
if image_source is not None:
    original = image_source

    # 1. RUN OPENCV BIOMETRIC TRACKER
    eye_count, is_microsleep, cv_annotated = analyze_eyes_opencv(original)

    # 2. RUN ROBOFLOW AI
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp_path = Path(temp.name)

    face_result, posture_result = None, None
    try:
        original.save(temp_path)
        with st.spinner("Running AI Diagnostics..."):
            try:
                face_result = client.run_workflow(workspace_name=WORKSPACE, workflow_id=FACE_WORKFLOW, images={"image": str(temp_path)}, use_cache=False)
                posture_result = client.run_workflow(workspace_name=WORKSPACE, workflow_id=POSTURE_WORKFLOW, images={"image": str(temp_path)}, parameters={"confidence": 0.4}, use_cache=False)
            except Exception: pass 
    finally:
        temp_path.unlink(missing_ok=True)

    face_boxes = get_boxes(face_result)
    posture_boxes = get_boxes(posture_result)
    accepted_face_boxes = [box for box in face_boxes if box.get("class") in {"awake", "drowsy"} and box.get("confidence", 0) >= FACE_THRESHOLD]
    head_on_wheel_boxes = [box for box in posture_boxes if box.get("class") == "head_on_wheel"]

    rf_annotated = original.copy()
    draw_boxes(rf_annotated, accepted_face_boxes, "orange")
    if head_on_wheel_boxes: draw_boxes(rf_annotated, head_on_wheel_boxes, "red")

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(rf_annotated, caption="Roboflow AI Detection", use_container_width=True)
    with col2:
        st.image(cv_annotated, caption="Biometric Eye Tracker (OpenCV)", use_container_width=True)

    st.markdown("---")
    st.write("### 📊 System Diagnostics")
    
    if eye_count is not None:
        if is_microsleep:
            st.error(f"🚨 **BIOMETRIC ALERT:** Micro-sleep detected! (0 Eyes Open).")
        else:
            st.success(f"✅ **BIOMETRIC STATUS:** Eyes open. ({eye_count} detected).")

    if head_on_wheel_boxes:
        st.error("🚨 **POSTURE ALERT:** Head-on-wheel detected. Check on driver immediately.")
    elif has_class(accepted_face_boxes, "drowsy"):
        st.warning("⚠️ **AI ALERT:** Drowsy appearance detected.")
    elif has_class(accepted_face_boxes, "awake") and not is_microsleep:
        st.info("ℹ️ **AI STATUS:** Driver is awake and alert.")

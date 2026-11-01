import os
import tempfile
from pathlib import Path
import math

import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
import cv2
import numpy as np

# --- SAFE MEDIAPIPE IMPORT ---
try:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

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

@st.cache_resource
def load_face_mesh():
    if not MP_AVAILABLE:
        return None
    return mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

client = load_client()
face_mesh = load_face_mesh()

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

# --- BIOMETRIC TRACKER ---
def analyze_biometrics(image):
    if not MP_AVAILABLE or face_mesh is None:
        return "MediaPipe library not loaded. Please check requirements.txt.", None

    img_rgb = np.array(image)
    results = face_mesh.process(img_rgb)
    
    if not results.multi_face_landmarks: 
        return "No face detected by biometric tracker.", None
        
    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = img_rgb.shape
    
    # Left and Right eye indices
    left_eye = [362, 385, 387, 263, 373, 380]
    right_eye = [33, 160, 158, 133, 153, 144]
    
    def get_ear(eye_indices):
        coords = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in eye_indices]
        v1 = math.dist(coords[1], coords[5])
        v2 = math.dist(coords[2], coords[4])
        h1 = math.dist(coords[0], coords[3])
        return (v1 + v2) / (2.0 * h1), coords

    left_ear, left_coords = get_ear(left_eye)
    right_ear, right_coords = get_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Draw eye dots
    img_annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for eye in [left_coords, right_coords]:
        for pt in eye:
            cv2.circle(img_annotated, pt, 2, (255, 255, 0), -1)
            
    img_out = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
    
    if avg_ear < 0.22:
        return f"🚨 MICRO-SLEEP DETECTED (EAR: {avg_ear:.2f})", img_out
    else:
        return f"✅ Eyes Open (EAR: {avg_ear:.2f})", img_out

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

    # 1. BIOMETRIC TRACKER
    biometric_msg, cv_annotated = analyze_biometrics(original)

    # 2. ROBOFLOW AI
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
        st.image(rf_annotated, caption="Roboflow AI (Face & Posture)", use_container_width=True)
    with col2:
        if cv_annotated is not None:
            st.image(cv_annotated, caption="Biometric Eye Tracker (MediaPipe)", use_container_width=True)

    st.markdown("---")
    st.write("### 📊 System Diagnostics")
    
    # Display Biometric Data
    if biometric_msg:
        if "MICRO-SLEEP" in biometric_msg:
            st.error(biometric_msg)
        else:
            st.success(biometric_msg)

    # Display AI Data
    if head_on_wheel_boxes:
        st.error("🚨 **POSTURE ALERT:** Head-on-wheel detected. Check on driver immediately.")
    elif has_class(accepted_face_boxes, "drowsy"):
        st.warning("⚠️ **AI ALERT:** Drowsy appearance detected.")
    elif has_class(accepted_face_boxes, "awake") and "MICRO-SLEEP" not in str(biometric_msg):
        st.info("ℹ️ **AI STATUS:** Driver is awake and alert.")

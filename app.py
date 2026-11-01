import os
import tempfile
from pathlib import Path

import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw

# --- PAGE SETUP ---
st.set_page_config(page_title="Orbital Guardian", page_icon="🛡️", layout="centered")

# 🚨 THE NEW CSS: Making the title MASSIVE
st.markdown("""
    <style>
    .main-title { 
        font-size: 56px; 
        font-weight: 900; 
        color: #0f172a; 
        margin-bottom: 0px; 
        line-height: 1.1;
    }
    .sub-title { 
        font-size: 20px; 
        color: #64748b; 
        margin-bottom: 30px; 
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🛡️ Orbital Guardian</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI Driver Monitoring & Fatigue Detection</p>', unsafe_allow_html=True)

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

client = load_client()

# --- TABS ---
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
    st.image(original, caption="Original photo", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp_path = Path(temp.name)

    face_result = None
    posture_result = None

    try:
        original.save(temp_path)

        with st.spinner("Analyzing driver..."):
            try:
                face_result = client.run_workflow(workspace_name=WORKSPACE, workflow_id=FACE_WORKFLOW, images={"image": str(temp_path)}, use_cache=False)
            except Exception: pass 

        with st.spinner("Checking posture..."):
            try:
                posture_result = client.run_workflow(workspace_name=WORKSPACE, workflow_id=POSTURE_WORKFLOW, images={"image": str(temp_path)}, parameters={"confidence": 0.4, "iou_threshold": 0.3, "class_agnostic_nms": False, "max_detections": 1000}, use_cache=False)
            except Exception: pass 
    finally:
        temp_path.unlink(missing_ok=True)

    face_boxes = get_boxes(face_result)
    posture_boxes = get_boxes(posture_result)

    accepted_face_boxes = [box for box in face_boxes if box.get("class") in {"awake", "drowsy"} and isinstance(box.get("confidence"), (int, float)) and box["confidence"] >= FACE_THRESHOLD]
    head_on_wheel_boxes = [box for box in posture_boxes if box.get("class") == "head_on_wheel"]

    annotated = original.copy()
    draw_boxes(annotated, accepted_face_boxes, "orange")
    
    if head_on_wheel_boxes:
        draw_boxes(annotated, head_on_wheel_boxes, "red")
        
    st.image(annotated, caption="Model detections", use_container_width=True)

    # --- THE VERDICT LOGIC ---
    st.markdown("---")
    st.write("### 📊 System Diagnostics")
    
    if head_on_wheel_boxes:
        st.error("⚠️ **DROWSINESS ALERT:** Head-on-wheel posture detected. Check on the driver immediately.")
    elif has_class(accepted_face_boxes, "drowsy"):
        st.warning("⚠️ **Possible drowsiness detected** from appearance.")
    elif has_class(accepted_face_boxes, "awake"):
        st.success("✅ **Awake appearance detected.**")
    else:
        st.info("ℹ️ **State unknown:** No confident face detected in this photo.")

    # --- THE ANALYTICS ---
    with st.expander("Raw Detection Details (Analytics)"):
        st.write("Face Data:")
        st.json(face_boxes)
        st.write("Posture Data:")
        st.json(posture_boxes)

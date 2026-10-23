import os
import tempfile
from pathlib import Path

import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw

WORKSPACE = "tsz-chung-pang"

# The version 1 RF-DETR Workflow that worked for your awake/drowsy photos.
FACE_WORKFLOW = (
    "drowsiness-detection-"
    "vdrowsiness-detection-b2qqq-iqk0f-1-rfdetr-small-t1-logic"
)

# The separate head-on-wheel posture Project.
POSTURE_WORKFLOW = "driver-head-on-wheel-posture"

FACE_THRESHOLD = 0.3

@st.cache_resource
def load_client():
    # HARDCODED API KEY
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="3zlKh7bl9yZXMl5RvxNY" 
    )

def get_boxes(result):
    if not isinstance(result, list) or not result:
        return []
    first = result[0]
    if not isinstance(first, dict):
        return []
    output = first.get("predictions", {})
    if isinstance(output, dict):
        boxes = output.get("predictions", [])
    elif isinstance(output, list):
        boxes = output
    else:
        boxes = []
    return [box for box in boxes if isinstance(box, dict)]

def has_class(boxes, class_name, minimum_confidence=0):
    return any(
        box.get("class") == class_name
        and isinstance(box.get("confidence"), (int, float))
        and box["confidence"] >= minimum_confidence
        for box in boxes
    )

def draw_boxes(image, boxes, color):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        try:
            x = float(box["x"])
            y = float(box["y"])
            width = float(box["width"])
            height = float(box["height"])
        except (KeyError, TypeError, ValueError):
            continue

        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2

        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text(
            (max(0, x1), max(0, y1 - 18)),
            str(box.get("class", "detection")),
            fill=color,
        )

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")
st.title("🚗 Driver Drowsiness Monitor")
st.write("Check a photo for drowsy appearance or head-on-wheel posture.")

client = load_client()
uploaded_file = st.file_uploader(
    "Choose a driver photo", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert("RGB")
    st.image(original, caption="Original photo", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp_path = Path(temp.name)

    face_result = None
    posture_result = None

    try:
        original.save(temp_path)

        # 1. Check Face
        with st.spinner("Analyzing driver..."):
            try:
                face_result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=FACE_WORKFLOW,
                    images={"image": str(temp_path)},
                    use_cache=False,
                )
            except Exception:
                pass # Silently ignore errors

        # 2. Check Posture (Silently)
        with st.spinner("Checking posture..."):
            try:
                posture_result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=POSTURE_WORKFLOW,
                    images={"image": str(temp_path)},
                    parameters={
                        "confidence": 0.4,
                        "iou_threshold": 0.3,
                        "class_agnostic_nms": False,
                        "max_detections": 1000,
                    },
                    use_cache=False,
                )
            except Exception:
                pass # Silently ignore errors if model is missing
    finally:
        temp_path.unlink(missing_ok=True)

    face_boxes = get_boxes(face_result)
    posture_boxes = get_boxes(posture_result)

    accepted_face_boxes = [
        box for box in face_boxes
        if box.get("class") in {"awake", "drowsy"}
        and isinstance(box.get("confidence"), (int, float))
        and box["confidence"] >= FACE_THRESHOLD
    ]
    head_on_wheel_boxes = [
        box for box in posture_boxes
        if box.get("class") == "head_on_wheel"
    ]

    annotated = original.copy()
    draw_boxes(annotated, accepted_face_boxes, "orange")
    
    # Only draw the red box if it actually found the posture
    if head_on_wheel_boxes:
        draw_boxes(annotated, head_on_wheel_boxes, "red")
        
    st.image(annotated, caption="Model detections", use_container_width=True)

    # --- THE NEAT VERDICT LOGIC ---
    
    # If Posture is detected, SHOW IT LOUDLY.
    if head_on_wheel_boxes:
        st.error("⚠️ DROWSINESS ALERT: Head-on-wheel posture detected. Check on the driver immediately.")
        
    # If Face is Drowsy, show it.
    elif has_class(accepted_face_boxes, "drowsy"):
        st.warning("⚠️ Possible drowsiness detected from appearance.")
        
    # If Face is Awake, show it.
    elif has_class(accepted_face_boxes, "awake"):
        st.info("✅ Awake appearance detected. No head-on-wheel posture detected.")
        
    # If nothing works, show a clean error (no mention of API failures).
    else:
        st.warning("⚠️ State unknown: no confident face or posture detected in this photo.")

    # The expander for debugging only
    with st.expander("Raw Detection Details"):
        st.write("Face Data:")
        st.json(face_boxes)
        st.write("Posture Data:")
        st.json(posture_boxes)

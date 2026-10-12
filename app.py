import streamlit as st
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")
st.title("🚗 Driver Drowsiness Monitor")
st.subheader("AI-Powered Fatigue Detection System")
st.write("Upload a driver's photo to detect drowsiness.")

@st.cache_resource
def load_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="3zlKh7bl9yZXMl5RvxNY"
    ).configure(InferenceConfiguration(api_key_transport="header"))

client = load_client()

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Fix transparency issues
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        try:
            result = client.run_workflow(
                workspace_name="tsz-chung-pang",
                workflow_id="drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-2-yolo26s-t1-logic",
                images={"image": temp_path},
                use_cache=True
            )
        except Exception as e:
            st.error(f"API Error: {e}")
            st.stop()

    # --- Extract predictions from different possible formats ---
    predictions = []
    if result and len(result) > 0:
        if isinstance(result[0], dict):
            preds_dict = result[0].get("predictions", {})
            if isinstance(preds_dict, dict):
                predictions = preds_dict.get("predictions", [])
            elif isinstance(preds_dict, list):
                predictions = preds_dict
        elif isinstance(result[0], list):
            predictions = result[0]

    # --- Filter: keep only predictions with confidence > 60% ---
    high_confidence_preds = [p for p in predictions if p.get("confidence", 0) > 0.6]

    # --- Display results ---
    if high_confidence_preds:
        first_pred_class = high_confidence_preds[0].get("class", "drowsy")
        
        if first_pred_class == "drowsy":
            st.error("⚠️ WARNING: Drowsiness detected!")
        else:
            st.success("✅ Driver is awake.")
        
        # Draw boxes and labels
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
            
        for pred in high_confidence_preds:
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            w = pred.get("width", 0)
            h = pred.get("height", 0)
            conf = pred.get("confidence", 0)
            cls = pred.get("class", "drowsy")
            
            color = "red" if cls == "drowsy" else "green"
            draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline=color, width=3)
            draw.text((x - w/2, y - h/2 - 30), f"{cls} {conf:.0%}", fill=color, font=font)
            
        st.image(image, caption='Detection Result', use_column_width=True)
        
        # --- Detailed Analysis ---
        st.markdown("---")
        st.markdown("### 📝 Detection Analysis")
        
        first_pred = high_confidence_preds[0]
        st.markdown(f"""
        Our AI has analyzed the image and detected a potential safety risk.
        
        | Metric | Value |
        | :--- | :--- |
        | **Detected Class** | `{first_pred.get('class', 'drowsy')}` |
        | **Confidence Score** | `{first_pred.get('confidence', 0):.2%}` |
        | **Bounding Box** | x: `{first_pred.get('x', 0):.1f}`, y: `{first_pred.get('y', 0):.1f}`, w: `{first_pred.get('width', 0):.1f}`, h: `{first_pred.get('height', 0):.1f}` |
        | **Detection Count** | `{len(high_confidence_preds)}` object(s) |
        
        **AI Decision Logic:**
        The model uses a YOLOv8 object detection algorithm. It analyzes the driver's facial features (eye closure, head position) within the bounding box. A confidence score above 60% indicates a high probability of drowsiness.
        
        **Recommendation:**
        🚨 The driver should pull over to a safe location and rest immediately.
        """)
        
    else:
        st.info("✅ No drowsiness detected.")
        st.markdown("""
        ---
        ### 📝 Detection Analysis
        Our AI found **no signs of drowsiness**. The driver appears to be alert.
        """)

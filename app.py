import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import time

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")
st.title("🚗 Driver Drowsiness Monitor")
st.subheader("AI-Powered Fatigue Detection System")
st.write("Upload a driver's photo to detect drowsiness.")

@st.cache_resource
def load_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="3zlKh7bl9yZXMl5RvxNY" 
    )

client = load_client()

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image (Original)', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        result = None
        for attempt in range(3):
            try:
                result = client.run_workflow(
                    workspace_name="tsz-chung-pang",
                    workflow_id="drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-2-yolo26s-t1-logic",
                    images={"image": temp_path},
                    use_cache=False
                )
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    st.error(f"API Error after 3 attempts: {e}")
                    st.stop()

    # --- RAW WORKFLOW RESPONSE ---
    with st.expander("Raw Workflow response"):
        st.json(result)

    # --- Extract predictions ---
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

    # --- Lower Confidence Threshold ---
    high_confidence_preds = [p for p in predictions if p.get("confidence", 0) > 0.3]

    # --- Display results ---
    if high_confidence_preds:
        first_pred = high_confidence_preds[0]
        cls = first_pred.get("class", "drowsy")
        conf = first_pred.get("confidence", 0)
        
        if cls == "drowsy":
            st.error("⚠️ WARNING: Drowsiness detected!")
        else:
            st.success("✅ Driver is awake.")
        
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
            c = pred.get("confidence", 0)
            cl = pred.get("class", "drowsy")
            
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            
            box_color = "red" if cl == "drowsy" else "green"
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            draw.text((x1, y1 - 30), f"{cl} {c:.0%}", fill=box_color, font=font)
            
        st.image(image, caption='Detection Result', use_column_width=True)
        
        st.markdown("---")
        st.markdown("### 📝 Detection Analysis")
        
        if cls == "drowsy":
            st.markdown(f"""
            Our AI has detected a potential safety risk.
            
            | Metric | Value |
            | :--- | :--- |
            | **Detected Class** | `drowsy` |
            | **Confidence Score** | `{conf:.2%}` |
            | **Detection Count** | `{len(high_confidence_preds)}` object(s) |
            
            **Recommendation:**
            🚨 The driver should pull over to a safe location and rest immediately.
            """)
        else:
            st.markdown(f"""
            Our AI has analyzed the image and found **no signs of drowsiness**.
            
            | Metric | Value |
            | :--- | :--- |
            | **Detected Class** | `awake` |
            | **Confidence Score** | `{conf:.2%}` |
            | **Detection Count** | `{len(high_confidence_preds)}` object(s) |
            
            **Recommendation:**
            ✅ The driver is alert. Continue driving safely.
            """)
        
    else:
        st.warning("⚠️ No detection result from the API. The AI could not find a face in this image.")
        st.markdown("""
        ---
        ### 📝 Detection Analysis
        The model returned an empty list `[]`. This means it did not recognize any faces in the image. This may be due to:
        - **The image is too wide/zoomed out** (the face is too small for the AI to see).
        - **The image angle or lighting conditions.**
        
        **Recommendation:** Please try uploading a closer, zoomed-in photo of the driver's face.
        """)

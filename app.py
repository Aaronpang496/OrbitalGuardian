import streamlit as st
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw

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
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Run the Roboflow workflow
        result = client.run_workflow(
            workspace_name="tsz-chung-pang",
            workflow_id="drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-2-yolo26s-t1-logic",
            images={"image": temp_path},
            use_cache=True
        )
        
    # --- Extract predictions from the workflow result ---
    predictions = []
    if result and len(result) > 0:
        predictions = result[0].get("predictions", {}).get("predictions", [])
    
    # --- Filter: only keep predictions with confidence > 60% ---
    high_confidence_preds = [p for p in predictions if p["confidence"] > 0.6]
    
    if high_confidence_preds:
        st.error("⚠️ WARNING: Drowsiness detected!")
        
        # Draw boxes and percentages on the image
        draw = ImageDraw.Draw(image)
        for pred in high_confidence_preds:
            x, y = pred["x"], pred["y"]
            w, h = pred["width"], pred["height"]
            conf, cls = pred["confidence"], pred["class"]
            
            # Draw the bounding box
            draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=3)
            # Draw the label
            draw.text((x - w/2, y - h/2 - 15), f"{cls} {conf:.0%}", fill="red")
            
        st.image(image, caption='Detection Result', use_column_width=True)
    else:
        st.info("✅ No drowsiness detected.")

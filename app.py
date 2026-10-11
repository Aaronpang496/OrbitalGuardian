import streamlit as st
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image, ImageDraw
import os

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")
st.title("🚗 Driver Drowsiness Monitor")
st.subheader("AI-Powered Fatigue Detection System")
st.write("Upload a driver's photo to detect drowsiness.")

@st.cache_resource
def load_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="3zlKh7bl9yZXMl5RvxNY"
    ).configure(InferenceConfiguration(
        api_key_transport="header"
    ))

client = load_client()

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        # Save the uploaded image to a temporary file
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Run the Roboflow workflow
        result = client.run_workflow(
            workspace_name="tsz-chung-pang",
            workflow_id="drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-2-yolo26s-t1-logic",
            images={"image": temp_path},
            use_cache=True
        )
        
    if result and len(result) > 0:
        predictions = result[0].get("predictions", {}).get("predictions", [])
        
        if predictions:
            st.error(f"⚠️ WARNING: Drowsiness detected!")
            
            # Draw boxes on the image
            draw = ImageDraw.Draw(image)
            for pred in predictions:
                x = pred["x"]
                y = pred["y"]
                w = pred["width"]
                h = pred["height"]
                draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=3)
            st.image(image, caption='Detection Result', use_column_width=True)
        else:
            st.info("✅ No drowsiness detected.")
    else:
        st.info("✅ No drowsiness detected.")

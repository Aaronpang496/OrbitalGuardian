import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")

st.title("🚗 Driver Drowsiness Monitor")
st.subheader("AI-Powered Fatigue Detection System")
st.write("Upload a driver's photo to detect drowsiness.")

@st.cache_resource
def load_model():
    # Initialize the Roboflow Inference client
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="YOUR_ROBOFLOW_API_KEY" # <--- 在這裡填上你的 Roboflow API Key
    )
    return client

client = load_model()

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        # Call the Roboflow model
        result = client.infer(image, model_id="drowsiness-detection-b2qqq/1")
        
    # Display the result
    st.json(result)
    
    if result.get("predictions"):
        st.error(f"⚠️ WARNING: {len(result['predictions'])} object(s) detected!")
    else:
        st.info("✅ No drowsiness detected.")

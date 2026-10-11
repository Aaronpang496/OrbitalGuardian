import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

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
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        result = client.infer(image, model_id="drowsiness-detection-b2qqq-iqk0f-1-rfdetr-small-t1")
        
    st.json(result)
    
    if result.get("predictions"):
        st.error(f"⚠️ WARNING: {len(result['predictions'])} object(s) detected!")
    else:
        st.info("✅ No drowsiness detected.")

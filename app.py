import streamlit as st
import requests
import base64
from PIL import Image
import io

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")
st.title("🚗 Driver Drowsiness Monitor")
st.subheader("AI-Powered Fatigue Detection System")
st.write("Upload a driver's photo to detect drowsiness.")

# Your Roboflow credentials
API_KEY = "3zlKh7bl9yZXMl5RvxNY"
MODEL_ID = "drowsiness-detection-b2qqq-iqk0f-1-rfdetr-small-t1"

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Call the Roboflow API
        url = f"https://serverless.roboflow.com/{MODEL_ID}/infer"
        params = {"api_key": API_KEY}
        
        response = requests.post(
            url,
            params=params,
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
    if response.status_code == 200:
        result = response.json()
        st.json(result)
        if result.get("predictions"):
            st.error(f"⚠️ WARNING: {len(result['predictions'])} object(s) detected!")
        else:
            st.info("✅ No drowsiness detected.")
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")

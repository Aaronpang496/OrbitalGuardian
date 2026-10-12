import streamlit as st
import requests
from PIL import Image, ImageDraw

st.set_page_config(page_title="Driver Drowsiness Monitor", page_icon="🚗")
st.title("🚗 Driver Drowsiness Monitor")
st.subheader("AI-Powered Fatigue Detection System")
st.write("Upload a driver's photo to detect drowsiness.")

# --- CONFIGURATION ---
API_KEY = "3zlKh7bl9yZXMl5RvxNY"
MODEL_ID = "drowsiness-detection-b2qqq-iqk0f-2-yolo26s-t1" # Your YOLO model ID

uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing...'):
        # Call the Roboflow Object Detection API with the image file
        url = f"https://detect.roboflow.com/{MODEL_ID}"
        params = {"api_key": API_KEY, "confidence": 30}
        
        # Send the image file directly
        response = requests.post(url, params=params, files={"file": uploaded_file.getvalue()})
        
    if response.status_code == 200:
        result = response.json()
        predictions = result.get("predictions", [])
        
        if predictions:
            st.error("⚠️ WARNING: Drowsiness detected!")
            
            # Draw boxes and labels
            draw = ImageDraw.Draw(image)
            for pred in predictions:
                x, y = pred["x"], pred["y"]
                w, h = pred["width"], pred["height"]
                conf, cls = pred["confidence"], pred["class"]
                
                # Draw the box
                draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=3)
                # Draw the label
                draw.text((x - w/2, y - h/2 - 15), f"{cls} {conf:.0%}", fill="red")
                
            st.image(image, caption='Detection Result', use_column_width=True)
        else:
            st.info("✅ No drowsiness detected.")
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")

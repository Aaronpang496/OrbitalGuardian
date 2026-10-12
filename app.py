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
        
        # Run the Roboflow workflow (the version that worked)
        result = client.run_workflow(
            workspace_name="tsz-chung-pang",
            workflow_id="drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-2-yolo26s-t1-logic",
            images={"image": temp_path},
            use_cache=True
        )
        
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

    # --- Display results ---
    if predictions:
        st.error("⚠️ WARNING: Drowsiness detected!")
        
        # Draw boxes and percentages
        draw = ImageDraw.Draw(image)
        for pred in predictions:
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            w = pred.get("width", 0)
            h = pred.get("height", 0)
            conf = pred.get("confidence", 0)
            cls = pred.get("class", "drowsy")
            
            draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=3)
            draw.text((x - w/2, y - h/2 - 15), f"{cls} {conf:.0%}", fill="red")
            
        st.image(image, caption='Detection Result', use_column_width=True)
        
        # Description
        st.markdown("""
        ---
        ### 📝 Detection Analysis
        Our AI has detected signs of **driver drowsiness**. 
        - **Recommendation:** The driver should pull over and rest immediately.
        """)
        
    else:
        st.info("✅ No drowsiness detected.")
        
        # Description
        st.markdown("""
        ---
        ### 📝 Detection Analysis
        Our AI found **no signs of drowsiness**. The driver appears to be alert.
        """)

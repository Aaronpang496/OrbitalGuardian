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
        
        # --- Run the Roboflow workflow ---
        result = client.run_workflow(
            workspace_name="tsz-chung-pang",
            workflow_id="drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-2-yolo26s-t1-logic",
            images={"image": temp_path},
            use_cache=True
        )

    # --- 顯示原始結果，讓我們看看 Roboflow 到底回傳了什麼 ---
    st.write("### 🔍 Raw API Response")
    st.json(result)

    # --- 嘗試抓取預測結果 ---
    predictions = []
    if result and len(result) > 0:
        # 情況 1: result[0] 是 dict，裡面有 "predictions"
        if isinstance(result[0], dict):
            if "predictions" in result[0]:
                preds = result[0]["predictions"]
                if isinstance(preds, dict) and "predictions" in preds:
                    predictions = preds["predictions"]
                elif isinstance(preds, list):
                    predictions = preds
        
        # 情況 2: result[0] 本身就是一個 list
        elif isinstance(result[0], list):
            predictions = result[0]

    # --- 畫框與顯示結果 ---
    if predictions:
        st.error("⚠️ WARNING: Drowsiness detected!")
        
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
        
        # --- 詳細描述 ---
        st.markdown("---")
        st.markdown("### 📝 Detection Analysis")
        first_pred = predictions[0]
        st.markdown(f"""
        Our AI has analyzed the image and detected a potential safety risk.
        
        | Metric | Value |
        | :--- | :--- |
        | **Detected Class** | `{first_pred.get('class', 'drowsy')}` |
        | **Confidence Score** | `{first_pred.get('confidence', 0):.2%}` |
        | **Bounding Box** | x: `{first_pred.get('x', 0):.1f}`, y: `{first_pred.get('y', 0):.1f}`, w: `{first_pred.get('width', 0):.1f}`, h: `{first_pred.get('height', 0):.1f}` |
        
        **AI Decision Logic:**
        The model uses a YOLOv8 object detection algorithm trained on the Drowsiness-Detection dataset.
        
        **Recommendation:**
        🚨 The driver should pull over to a safe location and rest immediately.
        """)
        
    else:
        st.info("✅ No drowsiness detected. (The model did not find any 'drowsy' objects in this image.)")

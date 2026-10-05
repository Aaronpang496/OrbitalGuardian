@st.cache_resource
def load_model():
    return YOLO('best.pt')

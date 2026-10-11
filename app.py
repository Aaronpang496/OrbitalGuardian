import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import gdown

@st.cache_resource
def load_model():
    url = 'https://drive.google.com/file/d/1yA1hcxG9Xn3CH_gG8QnmVws4gYnSBNU2/view?usp=share_link'
    output = 'best.pt'
    gdown.download(url, output, quiet=False)
    return YOLO(output)

pip install -U inference-sdk
# 1. Import the library
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="__ROBOFLOW_API_KEY__"
).configure(InferenceConfiguration(
    api_key_transport="header"  # header-based auth (inference v1.5.0+)
))

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="tsz-chung-pang",
    workflow_id="drowsiness-detection-vdrowsiness-detection-b2qqq-iqk0f-1-rfdetr-small-t1-logic",
    images={
        "image": "YOUR_IMAGE.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)

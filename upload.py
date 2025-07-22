from huggingface_hub import HfApi, upload_file

api = HfApi()

# 👇 Replace values as needed
model_path = "D:/saved_ml_model/model8.joblib"  # Your local model file
repo_id = "Ashwinharagi/surveillance-LightGBM"  # Your HF username/repo name

# Create repo (if not created)
api.create_repo(repo_id=repo_id, exist_ok=True)

# Upload the file
upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model8.joblib",  # How it will be named inside HF
    repo_id=repo_id,
    repo_type="model",
)

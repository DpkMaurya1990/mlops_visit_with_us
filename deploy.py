import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token=token)

api = HfApi()
space_id = "dpkmaurya2025/mlops-visit-with-us-v2" 

# README.md add kar diya hai list mein
files_to_upload = ["app.py", "Dockerfile", "requirements.txt", "README.md"]

for file in files_to_upload:
    if os.path.exists(file):
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=space_id,
            repo_type="space"
        )
        print(f"Uploaded {file}!")

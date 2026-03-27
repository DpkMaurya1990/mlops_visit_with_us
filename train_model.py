import os
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi, login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

api = HfApi()
repo_id = "dpkmaurya2025/mlops-visit-with-us-dataset" 
model_repo_id = "dpkmaurya2025/mlops-visit-with-us-model"

# MLflow Configuration
username = os.getenv("MLFLOW_TRACKING_USERNAME", "")
password = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")

os.environ['MLFLOW_TRACKING_USERNAME'] = username
os.environ['MLFLOW_TRACKING_PASSWORD'] = password

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Visit_With_Us_Experiment")

# Load data
train_path = hf_hub_download(repo_id=repo_id, filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=repo_id, filename="test.csv", repo_type="dataset")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Preprocessing
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# MLflow Training Block
with mlflow.start_run():
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')  #Adding class_weight to handle potential class imbalance in the dataset 
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))

    # Log to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(best_model, name="model")

    # Save metrics locally
    with open("metrics.json", "w") as f:
        json.dump({"best_params": grid_search.best_params_, "accuracy": acc}, f)

# Save and Upload
joblib.dump(best_model, "model.joblib") #joblib.dump actually overwrites the file if it already exists, so no need to check for existence
api.upload_file(path_or_fileobj="model.joblib", path_in_repo="model.joblib", repo_id=model_repo_id, repo_type="model")
api.upload_file(path_or_fileobj="metrics.json", path_in_repo="metrics.json", repo_id=model_repo_id, repo_type="model")

print("✅ Training and MLflow Logging Complete!")

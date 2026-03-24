import os
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi, login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token=token)

api = HfApi()
repo_id = "dpkmaurya2025/mlops-visit-with-us-dataset" 
model_repo_id = "dpkmaurya2025/mlops-visit-with-us-model"

# 1. Load Data
train_path = hf_hub_download(repo_id=repo_id, filename="train.csv", repo_type="dataset")
test_path = hf_hub_download(repo_id=repo_id, filename="test.csv", repo_type="dataset")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 2. Preprocessing & Encoding
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# 3. Define Model and Parameters for Tuning (Criteria-3 Requirement)
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# 4. Tune the Model (Grid Search)
print("Tuning model with GridSearchCV...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 5. Log all Tuned Parameters (Criteria-3 Requirement)
best_params = grid_search.best_params_
print(f"Best Parameters Found: {best_params}")

# Saving parameters to a JSON file for "Logging"
with open("metrics.json", "w") as f:
    json.dump({"best_params": best_params, "best_score": grid_search.best_score_}, f)

# 6. Evaluate Model Performance
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Final Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)

# 7. Register the BEST model in HF Model Hub
joblib.dump(best_model, "model.joblib")
api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="model.joblib", path_in_repo="model.joblib", repo_id=model_repo_id, repo_type="model")
# Also upload the metrics/logs
api.upload_file(path_or_fileobj="metrics.json", path_in_repo="metrics.json", repo_id=model_repo_id, repo_type="model")

print(f"Complete: Best model and logs registered at {model_repo_id}")

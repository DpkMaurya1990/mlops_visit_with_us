import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Visit With Us Predictor", layout="centered")

# Load the saved model from the Hugging Face model hub
@st.cache_resource
def load_model_from_hf():
    repo_id = "dpkmaurya2025/mlops-visit-with-us-model" 
    model_path = hf_hub_download(repo_id=repo_id, filename="model.joblib", repo_type="model")
    return joblib.load(model_path)

model = load_model_from_hf()

#All the features used in training the model 
FEATURES = [
    'Unnamed: 0','Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched',
    'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport',
    'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting',
    'Designation', 'MonthlyIncome'
]

st.title("🌲 Visit With Us: Wellness Package Predictor")

# 2. UI Inputs (Important ones for User)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 70, 30)
    income = st.number_input("Monthly Income", 0, 100000, 25000)
    passport = st.selectbox("Has Passport? (1=Yes, 0=No)", [0, 1])
with col2:
    trips = st.number_input("Number of Trips", 0, 10, 2)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# 3. Create a Full Feature DataFrame
if st.button("Predict Purchase"):
    # Dictionary with default values
    input_data = {feat: 0 for feat in FEATURES} # Start with 0 or average for all features

    #  Get the inputs and save them into a dataframe with the same structure as training data
    input_data.update({
        'Unnamed: 0': 0,
        'Age': age,
        'MonthlyIncome': income,
        'Passport': passport,
        'NumberOfTrips': trips,
        'PitchSatisfactionScore': satisfaction,
        'Designation': 0 if designation == "Executive" else 1 # Simple encoding match
    })

    # Convert to DataFrame with EXACT same column order as training
    input_df = pd.DataFrame([input_data])[FEATURES]

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Prediction: Customer is LIKELY to buy!")
    else:
        st.error("❌ Prediction: Customer is UNLIKELY to buy.")

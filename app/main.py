from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import shap

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Exact column order
MODEL_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Friendly names for display
FRIENDLY_NAMES = {
    'tenure': 'Customer Tenure',
    'MonthlyCharges': 'Monthly Charges',
    'TotalCharges': 'Total Charges',
    'SeniorCitizen': 'Senior Citizen',
    'gender': 'Gender',
    'Partner': 'Has Partner',
    'Dependents': 'Has Dependents',
    'PhoneService': 'Phone Service',
    'PaperlessBilling': 'Paperless Billing',
    'MultipleLines_No phone service': 'No Phone Service',
    'MultipleLines_Yes': 'Multiple Lines',
    'InternetService_Fiber optic': 'Fiber Optic Internet',
    'InternetService_No': 'No Internet Service',
    'OnlineSecurity_Yes': 'Online Security',
    'OnlineBackup_Yes': 'Online Backup',
    'DeviceProtection_Yes': 'Device Protection',
    'TechSupport_Yes': 'Tech Support',
    'StreamingTV_Yes': 'Streaming TV',
    'StreamingMovies_Yes': 'Streaming Movies',
    'Contract_One year': 'One Year Contract',
    'Contract_Two year': 'Two Year Contract',
    'PaymentMethod_Electronic check': 'Electronic Check Payment',
    'PaymentMethod_Mailed check': 'Mailed Check Payment',
    'PaymentMethod_Credit card (automatic)': 'Credit Card Payment',
}

# Initialize FastAPI
app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data structure
class CustomerData(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# Home route
@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running! ✅"}

# Prediction route
@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()

    # Rename columns
    renamed = {
        'MultipleLines_No_phone_service': 'MultipleLines_No phone service',
        'InternetService_Fiber_optic': 'InternetService_Fiber optic',
        'OnlineSecurity_No_internet_service': 'OnlineSecurity_No internet service',
        'OnlineBackup_No_internet_service': 'OnlineBackup_No internet service',
        'DeviceProtection_No_internet_service': 'DeviceProtection_No internet service',
        'TechSupport_No_internet_service': 'TechSupport_No internet service',
        'StreamingTV_No_internet_service': 'StreamingTV_No internet service',
        'StreamingMovies_No_internet_service': 'StreamingMovies_No internet service',
        'Contract_One_year': 'Contract_One year',
        'Contract_Two_year': 'Contract_Two year',
        'PaymentMethod_Credit_card_automatic': 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check'
    }

    for old_key, new_key in renamed.items():
        input_dict[new_key] = input_dict.pop(old_key)

    # Create dataframe
    input_data = pd.DataFrame([input_dict])
    input_data = input_data[MODEL_COLUMNS]

    # Predict
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # SHAP explanation
    shap_values = explainer.shap_values(input_data)
    shap_series = pd.Series(shap_values[0], index=MODEL_COLUMNS)

    # Top 3 reasons pushing towards churn (positive SHAP)
    churn_reasons = shap_series[shap_series > 0]\
        .sort_values(ascending=False)\
        .head(3)

    # Top 3 reasons pushing away from churn (negative SHAP)
    safe_reasons = shap_series[shap_series < 0]\
        .sort_values(ascending=True)\
        .head(3)

    # Format reasons
    def format_reasons(reasons):
        result = []
        for feature, value in reasons.items():
            friendly = FRIENDLY_NAMES.get(feature, feature)
            result.append({
                "feature": friendly,
                "impact": round(float(value), 3)
            })
        return result

    return {
        "churn_prediction": int(prediction[0]),
        "churn_meaning": "Will Churn" if prediction[0] == 1 else "Will Stay",
        "churn_probability": round(float(probability[0][1]) * 100, 2),
        "churn_reasons": format_reasons(churn_reasons),
        "safe_reasons": format_reasons(safe_reasons)
    }
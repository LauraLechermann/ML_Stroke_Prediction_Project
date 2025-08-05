import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


class FeatureSelector:
    def __init__(self, indices):
        self.indices = indices
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[:, self.indices]


# 1. Load Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_model_artifacts/final_model.pkl")
    pipeline = joblib.load("final_model_artifacts/preprocessing_pipeline.pkl")
    with open("final_model_artifacts/feature_names.json") as f:
        feature_names = json.load(f)
    return model, pipeline, feature_names

model, pipeline, feature_names = load_artifacts()

# 2. Feature Engineering Class
class StrokeFeatureEngineering:
    def transform(self, df):
        df = df.copy()

        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 65, 120],
                                 labels=['young', 'senior'])
 
        df['bmi_category'] = pd.cut(df['bmi'].fillna(28.0), 
                                    bins=[0, 18.5, 25, 30, 100],
                                    labels=['underweight', 'normal', 'overweight', 'obese'])  

        df['glucose_category'] = pd.cut(df['avg_glucose_level'],
                                        bins=[0, 100, 126, 300],  
                                        labels=['normal', 'prediabetic', 'very_high']) 

        df['risk_factor_count'] = (
            (df['hypertension'] == 1).astype(int) +
            (df['heart_disease'] == 1).astype(int) +
            (df['smoking_status'].isin(['smokes', 'formerly smoked'])).astype(int) +
            (df['age'] > 65).astype(int) +
            (df['avg_glucose_level'] > 126).astype(int)
        )
        df['bmi_glucose_ratio'] = df['bmi'].fillna(28.0) / df['avg_glucose_level']
        df['bmi_missing'] = df['bmi'].isna().astype(int)
        df['is_senior'] = (df['age'] >= 65).astype(int)
        df['high_risk_group'] = ((df['age'] > 65) & 
                                ((df['hypertension'] == 1) | 
                                 (df['heart_disease'] == 1) | 
                                 (df['avg_glucose_level'] > 126))).astype(int)
        return df


        
# 3. Prediction function
def predict_stroke(patient_input: dict):
    df = pd.DataFrame([patient_input])
    df_engineered = StrokeFeatureEngineering().transform(df)
    X = pipeline.transform(df_engineered)
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    
    if prob < 0.1:
        risk_level = "Low"
        label = "Low Stroke Risk"
    elif prob < 0.3:
        risk_level = "Moderate"
        label = "Low Stroke Risk"
    elif prob < 0.5:
        risk_level = "High"
        label = "Stroke Risk Detected"  # ✅ FIXED!
    else:
        risk_level = "Very High"
        label = "Stroke Risk Detected"
    
    return {
        "prediction": int(pred),
        "probability": round(prob, 3),
        "risk_level": risk_level,
        "label": label  # ✅ FIXED: Use consistent logic!
    }


    

# 4. Streamlit UI
st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="wide")
st.title("🧠 Stroke Risk Assessment Tool")

st.markdown("Provide patient data to estimate the risk of stroke.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
with col2:
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    glucose = st.number_input("Avg Glucose Level", value=100.0)
    bmi = st.number_input("BMI", value=25.0)
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

if st.button("🔍 Predict Stroke Risk"):
    input_dict = {
        'gender': gender,
        'age': age,
        'hypertension': int(hypertension == "Yes"),
        'heart_disease': int(heart_disease == "Yes"),
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence,
        'avg_glucose_level': glucose,
        'bmi': bmi,
        'smoking_status': smoking
    }
    result = predict_stroke(input_dict)

    st.subheader("📊 Prediction Result")
    st.write(f"**Prediction:** {result['label']}")
    st.write(f"**Probability of Stroke:** {result['probability']:.1%}")
    st.write(f"**Risk Level:** {result['risk_level']}")

st.markdown("---")
st.caption("⚠️ This tool is for educational use only and not a substitute for professional diagnosis.")

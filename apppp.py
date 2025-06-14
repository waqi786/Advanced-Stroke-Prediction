import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and transformers
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("stroke_scaler.pkl")
encoder = joblib.load("stroke_encoder.pkl")

st.set_page_config(page_title="Stroke Prediction", layout="wide")
st.title("🧠 Stroke Risk Prediction App ( HealthCare Data )")
st.markdown("---")
st.markdown("### 📌 Introduction")
st.markdown("""
This interactive web application helps you assess your risk of having a stroke based on common health and lifestyle factors.  
It uses a **Logistic Regression Machine Learning model**, trained on real healthcare data, to make predictions.

🧠 A **stroke** is a serious medical condition that occurs when blood flow to part of the brain is interrupted.  
Early detection and prevention are critical — especially for individuals with high blood pressure, diabetes, or smoking habits.

### 🔍 How it works:
- Enter your personal health information such as age, BMI, glucose level, smoking status, etc.
- The model processes your inputs and predicts whether you are at **high risk** or **low risk** of stroke.
- You'll receive an immediate result based on your data.""")

st.markdown("### ⚙️ Input User Details")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 0, 120, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Prepare input DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [Residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Encode categoricals
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    input_data[col] = encoder[col].transform(input_data[col])

# Scale numerical
input_data[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(input_data[['age', 'avg_glucose_level', 'bmi']])

# Predict
if st.button("🔍 Predict Stroke Risk"):
    prediction = model.predict(input_data)[0]
    st.subheader("🧾 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ High risk of stroke detected!")
    elif prediction == 0:
        st.error("⚠️ Low risk of stroke detected!")
    else:
        st.success("✅ No stroke risk detected.")

st.markdown("---")
st.markdown("### 📌 Conclusion")
st.markdown("This logistic regression model helps identify individuals at risk of stroke. Key indicators include glucose levels, age, and hypertension.")
st.caption("Created By | WAQAR ALI")

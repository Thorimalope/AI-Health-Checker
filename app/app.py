import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="AI Health Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Health Predictor")
st.write("Enter your health details below to estimate diabetes risk.")

# -------------------------
# Load model + scaler
# -------------------------
@st.cache_resource
def load_artifacts():
    with open("models/baseline_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_artifacts()

# -------------------------
# User inputs
# -------------------------
st.subheader("Patient Inputs")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=120, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=32.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

input_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
}

# -------------------------
# Predict button
# -------------------------
st.subheader("Prediction")

if st.button("Predict Risk"):
    input_df = pd.DataFrame([input_data])

    # Scale inputs
    input_scaled = scaler.transform(input_df)

    # Predict probability
    proba = model.predict_proba(input_scaled)[0, 1]
    pred = int(proba >= 0.5)

    # Risk banding (simple)
    if proba < 0.33:
        risk_level = "Low"
    elif proba < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    st.write(f"**Diabetes Risk Probability:** `{proba:.2f}`")
    st.write(f"**Risk Level:** **{risk_level}**")

    if pred == 1:
        st.warning("Model suggests higher diabetes risk. Consider consulting a medical professional.")
    else:
        st.success("Model suggests lower diabetes risk. Maintain healthy habits and regular checkups.")

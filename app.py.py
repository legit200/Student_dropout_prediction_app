import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("xgb_top6_model.pkl")
scaler = joblib.load("scaler_top6.pkl")

# Title and description
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("🎓 Student Dropout Prediction App")
st.write("This app predicts whether a student is likely to **Graduate** or **Dropout** based on key academic features.")

# Input form
with st.form("prediction_form"):
    st.subheader("📋 Enter Student Details")

    feature1 = st.number_input("Curricular units 2nd sem (approved)", min_value=0.0, format="%.2f")
    feature2 = st.number_input("Curricular units 1st sem (approved)", min_value=0.0, format="%.2f")
    feature3 = st.number_input("Curricular units 2nd sem (grade)", min_value=0.0, format="%.2f")
    feature4 = st.number_input("Curricular units 1st sem (grade)", min_value=0.0, format="%.2f")
    feature5 = st.selectbox("Tuition fees up to date", ["No (0)", "Yes (1)"])
    feature6 = st.selectbox("Scholarship holder", ["No (0)", "Yes (1)"])

    submitted = st.form_submit_button("🔍 Predict")

# Prediction logic
if submitted:
    try:
        features = [
            feature1,
            feature2,
            feature3,
            feature4,
            int(feature5.split()[1][1]),
            int(feature6.split()[1][1])
        ]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba([features_scaled[0]])[0][1]

        if prediction == 1:
            st.success(f"✅ The student is likely to **Graduate**.")
        else:
            st.error(f"⚠️ The student is at risk of **Dropping Out**.")

        st.markdown(f"**📈 Probability of Graduation:** `{proba*100:.2f}%`")
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")

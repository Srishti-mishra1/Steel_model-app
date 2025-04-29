import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("Steel_model.pkl")

st.set_page_config(page_title="Steel Fatigue Strength Predictor", layout="centered")
st.title("🔩 Steel Fatigue Strength Predictor")
st.markdown("""
This app predicts the **Fatigue Strength (MPa)** of steel using its chemical composition and processing parameters.
""")

# Input fields
with st.form("input_form"):
    st.subheader("📥 Enter Material Properties")
    c_pct = st.number_input("Carbon (%)", 0.00, 2.00, step=0.01, value=0.25)
    mn_pct = st.number_input("Manganese (%)", 0.00, 2.00, step=0.01, value=1.20)
    cr_pct = st.number_input("Chromium (%)", 0.00, 2.00, step=0.01, value=0.60)
    ni_pct = st.number_input("Nickel (%)", 0.00, 2.00, step=0.01, value=0.30)
    mo_pct = st.number_input("Molybdenum (%)", 0.00, 2.00, step=0.01, value=0.10)
    cu_pct = st.number_input("Copper (%)", 0.00, 2.00, step=0.01, value=0.20)
    temperature = st.number_input("Heat Treatment Temp (°C)", 600, 1200, step=10, value=850)
    cooling_rate = st.number_input("Cooling Rate (°C/s)", 0.0, 10.0, step=0.1, value=0.5)

    submitted = st.form_submit_button("🔍 Predict Fatigue Strength")

if submitted:
    # Prepare input
    input_data = pd.DataFrame([{
        'C (%)': c_pct,
        'Mn (%)': mn_pct,
        'Cr (%)': cr_pct,
        'Ni (%)': ni_pct,
        'Mo (%)': mo_pct,
        'Cu (%)': cu_pct,
        'Heat Treatment Temp (°C)': temperature,
        'Cooling Rate (°C/s)': cooling_rate
    }])

    # Ensure columns match exactly
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Fatigue Strength: **{prediction:.2f} MPa**")
    except ValueError as e:
        st.error(f"❌ Prediction failed: {e}")
        st.write("Ensure your input feature names exactly match the trained model's columns:")
        st.write("Expected:", model.feature_names_in_.tolist())
        st.write("Received:", input_data.columns.tolist())

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and XGBoost.")

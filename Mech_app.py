import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("Steel_model.pkl")

st.title("Steel Fatigue Strength Predictor")

# Input fields
c_pct = st.number_input("Carbon (%)", 0.0, 2.0, 0.25)
mn_pct = st.number_input("Manganese (%)", 0.0, 2.0, 1.2)
cr_pct = st.number_input("Chromium (%)", 0.0, 2.0, 0.6)
ni_pct = st.number_input("Nickel (%)", 0.0, 2.0, 0.3)
mo_pct = st.number_input("Molybdenum (%)", 0.0, 2.0, 0.1)
cu_pct = st.number_input("Copper (%)", 0.0, 2.0, 0.2)
temperature = st.number_input("Heat Treatment Temp (°C)", 600, 1200, 850)
cooling_rate = st.number_input("Cooling Rate (°C/s)", 0.0, 10.0, 0.5)

if st.button("Predict Fatigue Strength"):
    # Create a DataFrame for prediction
    input_df = pd.DataFrame([{
        'C (%)': c_pct,
        'Mn (%)': mn_pct,
        'Cr (%)': cr_pct,
        'Ni (%)': ni_pct,
        'Mo (%)': mo_pct,
        'Cu (%)': cu_pct,
        'Heat Treatment Temp (°C)': temperature,
        'Cooling Rate (°C/s)': cooling_rate
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Fatigue Strength: {prediction:.2f} MPa")

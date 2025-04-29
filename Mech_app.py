import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("Steel_model.pkl")

st.set_page_config(page_title="Steel Fatigue Strength Predictor", layout="wide")
st.title("🔩 Steel Fatigue Strength Predictor (27 Features)")
st.markdown("This app predicts **Fatigue Strength (MPa)** of steel based on detailed compositional, process, and mechanical inputs.")

# Create form for input
with st.form("input_form"):
    st.subheader("🧪 Composition Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        c_pct = st.number_input("C (%)", 0.00, 2.00, 0.25)
        si_pct = st.number_input("Si (%)", 0.00, 2.00, 0.30)
        mn_pct = st.number_input("Mn (%)", 0.00, 2.00, 1.20)
        p_pct = st.number_input("P (%)", 0.000, 1.000, 0.030)
        s_pct = st.number_input("S (%)", 0.000, 1.000, 0.020)
        ni_pct = st.number_input("Ni (%)", 0.00, 2.00, 0.30)
        cr_pct = st.number_input("Cr (%)", 0.00, 2.00, 0.60)
        cu_pct = st.number_input("Cu (%)", 0.00, 2.00, 0.20)
        mo_pct = st.number_input("Mo (%)", 0.00, 2.00, 0.10)

    with col2:
        ingot_size = st.number_input("Ingot Size (kg)", 100, 10000, 500)
        reduction_ratio = st.number_input("Reduction Ratio (%)", 0.0, 100.0, 70.0)
        inclusion_rating = st.number_input("Inclusion Rating", 0.0, 5.0, 1.5)
        normalizing_temp = st.number_input("Normalizing Temp (°C)", 600, 1100, 850)
        normalizing_time = st.number_input("Normalizing Time (h)", 0.5, 10.0, 2.0)
        quenching_temp = st.number_input("Quenching Temp (°C)", 600, 1100, 900)
        quenching_time = st.number_input("Quenching Time (s)", 10, 10000, 120)

    with col3:
        tempering_temp = st.number_input("Tempering Temp (°C)", 200, 800, 500)
        tempering_time = st.number_input("Tempering Time (h)", 0.5, 10.0, 2.0)
        ys = st.number_input("Yield Strength (MPa)", 0, 2000, 450)
        uts = st.number_input("Ultimate Tensile Strength (MPa)", 0, 2000, 600)
        elong = st.number_input("% Elongation", 0.0, 100.0, 20.0)
        red_area = st.number_input("% Reduction in Area", 0.0, 100.0, 30.0)
        hardness = st.number_input("Hardness (HV)", 50, 500, 180)
        impact_energy = st.number_input("Charpy Impact Energy (J)", 0, 300, 80)

    st.subheader("🧠 Derived & Categorical Inputs")
    ceq = st.number_input("Carbon Equivalent (Ceq)", 0.2, 2.0, 0.65)
    high_carbon = st.selectbox("High Carbon (1 = Yes, 0 = No)", [0, 1])
    high_alloy = st.selectbox("High Alloy (1 = Yes, 0 = No)", [0, 1])

    submitted = st.form_submit_button("🔍 Predict Fatigue Strength")

# Make prediction
if submitted:
    input_data = pd.DataFrame([{
        'C (%)': c_pct,
        'Si (%)': si_pct,
        'Mn (%)': mn_pct,
        'P (%)': p_pct,
        'S (%)': s_pct,
        'Ni (%)': ni_pct,
        'Cr (%)': cr_pct,
        'Cu (%)': cu_pct,
        'Mo (%)': mo_pct,
        'Ingot Size (kg)': ingot_size,
        'Reduction Ratio (%)': reduction_ratio,
        'Inclusion Rating': inclusion_rating,
        'Normalizing Temp (°C)': normalizing_temp,
        'Normalizing Time (h)': normalizing_time,
        'Quenching Temp (°C)': quenching_temp,
        'Quenching Time (s)': quenching_time,
        'Tempering Temp (°C)': tempering_temp,
        'Tempering Time (h)': tempering_time,
        'Yield Strength (MPa)': ys,
        'Ultimate Tensile Strength (MPa)': uts,
        '% Elongation': elong,
        '% Reduction in Area': red_area,
        'Hardness (HV)': hardness,
        'Charpy Impact Energy (J)': impact_energy,
        'Ceq': ceq,
        'High Carbon': high_carbon,
        'High Alloy': high_alloy
    }])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"✅ Predicted Fatigue Strength: **{prediction:.2f} MPa**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.write("Expected features:", model.feature_names_in_.tolist())
        st.write("Received:", input_data.columns.tolist())


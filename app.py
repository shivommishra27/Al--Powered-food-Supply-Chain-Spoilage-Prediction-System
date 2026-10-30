"""Streamlit dashboard.  Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.config import FEATURES, PACKAGING, PRODUCTS
from src.predict import predict_batch, predict_one

st.set_page_config(page_title="Food Spoilage Predictor", page_icon="🥦", layout="wide")
st.title("🥦 AI-Powered Food Supply Chain Spoilage Prediction")
st.caption("Predict spoilage risk and remaining shelf life from cold-chain sensor data.")

tab1, tab2 = st.tabs(["Single shipment", "Batch (CSV)"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        product = st.selectbox("Product", list(PRODUCTS))
        packaging = st.selectbox("Packaging", list(PACKAGING))
        days = st.slider("Days since harvest", 0.0, 60.0, 2.0, 0.5)
        transit = st.slider("Transit hours", 0.0, 150.0, 30.0, 1.0)
    with c2:
        temp = st.slider("Average temperature (°C)", -5.0, 35.0, float(PRODUCTS[product][1]) + 3, 0.5)
        tstd = st.slider("Temperature std dev (°C)", 0.0, 6.0, 1.0, 0.1)
        hum = st.slider("Humidity (%)", 30, 100, 80)
    with c3:
        co2 = st.slider("CO₂ (ppm)", 300, 2500, 600, 50)
        eth = st.slider("Ethylene (ppm)", 0.0, 40.0, 5.0, 0.5)
        doors = st.slider("Door-open events", 0, 20, 3)
        vib = st.slider("Vibration (g)", 0.0, 2.0, 0.4, 0.05)

    if st.button("Predict", type="primary"):
        record = dict(product=product, packaging=packaging, avg_temp_c=temp,
                      temp_std_c=tstd, humidity_pct=hum, transit_hours=transit,
                      days_since_harvest=days, co2_ppm=co2, ethylene_ppm=eth,
                      door_open_events=doors, vibration_g=vib)
        try:
            res = predict_one(record)
        except FileNotFoundError as e:
            st.error(str(e))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Spoilage probability", f"{res['spoilage_probability']:.1%}")
            m2.metric("Remaining shelf life", f"{res['remaining_shelf_life_days']:.1f} days")
            m3.metric("Risk level", res["risk_level"])
            {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[res["risk_level"]](res["recommendation"])

with tab2:
    st.write("Upload a CSV with columns:", ", ".join(FEATURES))
    file = st.file_uploader("CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            out = predict_batch(df)
            st.dataframe(out)
            st.bar_chart(out["risk_level"].value_counts())
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")

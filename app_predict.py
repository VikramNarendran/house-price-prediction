import streamlit as st
import pandas as pd
from utils.predict import predict_price

# ─────────────────────────────────────────────────────────────
# 🎯 Page Setup
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")
st.markdown("Estimate house prices using a trained ML model. Choose manual input for a single prediction or upload a CSV for batch predictions.")

# ─────────────────────────────────────────────────────────────
# 🔧 Manual Prediction
st.header("🔍 Manual Input")

with st.form("manual_form"):
    col1, col2 = st.columns(2)
    with col1:
        crime_rate = st.number_input("Crime Rate", min_value=0.0, value=0.00632)
        resid_area = st.number_input("Residential Area", min_value=0.0, value=32.21)
        air_qual = st.number_input("Air Quality", min_value=0.0)
        room_num = st.number_input("Number of Rooms", min_value=0.0)
        age = st.number_input("House Age", min_value=0.0)
        teachers = st.number_input("Teachers Nearby", min_value=0.0)
    with col2:
        poor_prop = st.number_input("Poor Population %", min_value=0.0)
        n_hos_beds = st.number_input("Hospital Beds Nearby", min_value=0.0)
        parks = st.number_input("Parks Nearby", min_value=0.0)
        dist = st.number_input("Average Distance", min_value=0.0)
        airport = st.selectbox("Airport Nearby?", ["NO", "YES"])
        waterbody_River = st.selectbox("River Nearby?", ["NO", "YES"])

    submitted = st.form_submit_button("💡 Predict Price")
    if submitted:
        raw_features = [
            crime_rate, resid_area, air_qual, room_num, age,
            teachers, poor_prop, n_hos_beds, parks, dist,
            1 if airport == "YES" else 0,
            1 if waterbody_River == "YES" else 0
        ]
        try:
            price = predict_price(raw_features)
            st.success(f"💰 Estimated House Price: **${price:,.2f} million**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ─────────────────────────────────────────────────────────────
# 📁 Batch Prediction via CSV
st.header("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV with input parameters", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Convert YES/NO to 1/0 if needed
        if df["airport"].dtype == object:
            df["airport"] = df["airport"].map({"YES": 1, "NO": 0})
        if df["waterbody_River"].dtype == object:
            df["waterbody_River"] = df["waterbody_River"].map({"YES": 1, "NO": 0})

        expected_cols = [
            "crime_rate", "resid_area", "air_qual", "room_num", "age",
            "teachers", "poor_prop", "n_hos_beds", "parks", "dist",
            "airport", "waterbody_River"
        ]

        # Validate and clean
        df[expected_cols] = df[expected_cols].apply(pd.to_numeric, errors='coerce')
        invalid_rows = df[df[expected_cols].isnull().any(axis=1)]
        df = df.dropna(subset=expected_cols)

        if not invalid_rows.empty:
            st.warning(f"⚠️ {len(invalid_rows)} row(s) skipped due to invalid values.")

        # Run predictions
        df["Predicted_Price"] = [
            predict_price([float(row[col]) for col in expected_cols])
            for _, row in df.iterrows()
        ]

        # Reset index to start from 0
        df.reset_index(drop=True, inplace=True)

        st.success("✅ Predictions complete.")
        st.subheader("📊 Predicted Prices (in USD millions)")
        st.dataframe(df[["Predicted_Price"]].style.format({"Predicted_Price": "${:,.2f}M"}))

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Results (USD millions)",
            data=csv,
            file_name="predicted_prices.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Failed to process file: {e}")
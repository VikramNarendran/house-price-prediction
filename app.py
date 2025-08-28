import streamlit as st
from utils.predict import predict_price

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")

st.markdown("Enter the details below to estimate the house price using a Box-Cox transformed regression model.")

# Input fields
crime_rate = st.number_input("Crime Rate", min_value=0.0, value=0.00632)
resid_area = st.number_input("Residential Area", min_value=0.0, value=32.21)
air_qual = st.number_input("Air Quality", min_value=0.0)
room_num = st.number_input("Number of Rooms", min_value=0.0)
age = st.number_input("House Age", min_value=0.0)
teachers = st.number_input("Number of Teachers", min_value=0.0)
poor_prop = st.number_input("Poor Population %", min_value=0.0)
n_hos_beds = st.number_input("Hospital Beds Nearby", min_value=0.0)
parks = st.number_input("Number of Parks Nearby", min_value=0.0)
dist = st.number_input("Average Distance", min_value=0.0)
waterbody_River = int(st.checkbox("Is there a River nearby?"))
airport = int(st.checkbox("Is there an Airport nearby?"))

# Predict button
if st.button("Predict Price"):
    raw_features = [
        crime_rate, resid_area, air_qual, room_num, age,
        teachers, poor_prop, n_hos_beds, parks, dist,
        airport, waterbody_River
    ]

    try:
        with st.spinner("Predicting..."):
            price = predict_price(raw_features)
        st.success(f"💰 Estimated House Price: {price:.2f}")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
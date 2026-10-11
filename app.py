
import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Uber Fare Predictor",
    page_icon="🚕",
    layout="centered"
)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load("D:\\EDA projects\\Uber-Analysis\\models\\best_model.pkl")

model = load_model()

# 3. App Title
st.title("🚕 Uber Fare Predictor")
st.write("Enter your trip details to estimate the fare.")

# 4. User Inputs
distance = st.number_input(
    "Trip Distance (km)",
    min_value=0.1,
    max_value=500.0,
    value=5.0,
    step=0.5
)

passengers = st.number_input(
    "Passenger Count",
    min_value=1,
    max_value=6,
    value=1,
    step=1
)

hour = st.slider(
    "Pickup Hour",
    min_value=0,
    max_value=23,
    value=12
)

day = st.slider(
    "Pickup Day",
    min_value=1,
    max_value=31,
    value=15
)

month = st.selectbox(
    "Pickup Month",
    options=list(range(1, 13)),
    index=0
)

day_of_week = st.selectbox(
    "Day of Week",
    options=[
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
)

# 5. Prediction
if st.button("Predict Fare"):
    day_number = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ].index(day_of_week)

    input_data = pd.DataFrame([{
        "distance_km": distance,
        "passenger_count": passengers,
        "pickup_hour": hour,
        "pickup_day": day,
        "pickup_month": month,
        "pickup_dayofweek": day_number
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Fare: ${prediction:.2f}")
    st.caption("This is a model estimate, not an official Uber quote.")

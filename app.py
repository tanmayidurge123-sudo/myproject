import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Title
# -----------------------------
st.title("Insurance Customer Response Prediction")

# -----------------------------
# User Input
# -----------------------------
st.header("Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
driving_license = st.selectbox("Driving License (0 = No, 1 = Yes)", [0, 1])
region_code = st.number_input("Region Code", value=28)
previously_insured = st.selectbox("Previously Insured (0 = No, 1 = Yes)", [0, 1])
vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
vehicle_damage = st.selectbox("Vehicle Damage", ["No", "Yes"])
annual_premium = st.number_input("Annual Premium", value=30000)
policy_sales_channel = st.number_input("Policy Sales Channel", value=26)
vintage = st.number_input("Vintage (Days)", value=100)

# -----------------------------
# Convert categorical inputs
# -----------------------------
vehicle_age_map = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
vehicle_damage_map = {"No": 0, "Yes": 1}

vehicle_age = vehicle_age_map[vehicle_age]
vehicle_damage = vehicle_damage_map[vehicle_damage]

# -----------------------------
# Dummy Model (Replace with your trained model)
# -----------------------------
# NOTE: In real project, you should load your trained model
rf = RandomForestClassifier()

# Dummy training (for demo purpose only)
X_dummy = np.random.rand(100, 9)
y_dummy = np.random.randint(0, 2, 100)
rf.fit(X_dummy, y_dummy)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
scaler.fit(X_dummy)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    user_data = np.array([[age, driving_license, region_code,
                           previously_insured, vehicle_age,
                           vehicle_damage, annual_premium,
                           policy_sales_channel, vintage]])

    user_data = scaler.transform(user_data)

    prediction = rf.predict(user_data)[0]

    if prediction == 1:
        st.success("Customer WILL respond")
    else:
        st.error("Customer will NOT respond")
import streamlit as st
import requests

st.title("Printer Failure Prediction Dashboard")

temperature = st.slider("Temperature", 30, 100)
voltage = st.slider("Voltage", 200, 250)
print_count = st.slider("Print Count", 1000, 50000)
error_count = st.slider("Error Count", 0, 20)
maintenance_gap = st.slider("Maintenance Gap", 1, 365)

if st.button("Predict"):
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={
            "temperature": temperature,
            "voltage": voltage,
            "print_count": print_count,
            "error_count": error_count,
            "maintenance_gap": maintenance_gap
        }
    )

    st.write(response.json())

import joblib
import pandas as pd
import streamlit as st

# Saved model aur columns load karein
model = joblib.load("car_loan_model.pkl")  # Agar model ka naam kuch aur hai toh yahan change kar lein
model_columns = joblib.load("model_columns.pkl")

st.title("🚗 Car Loan EMI Prediction App")
st.write("Apni car loan ki details niche bharein:")

# Car Loan dataset ke mutabiq inputs (Apne notebook ke columns ke hisab se inhe set karein)
car_price = st.number_input("Car Price (INR)", min_value=0, value=500000)
down_payment = st.number_input("Down Payment (INR)", min_value=0, value=100000)
loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=12, value=60)
interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, value=8.5)
monthly_income = st.number_input("Monthly Income (INR)", min_value=0, value=50000)

if st.button("Predict EMI"):
    # Input data ko DataFrame me convert karein
    input_data = pd.DataFrame({
        "Car_Price_INR": [car_price],
        "Down_Payment_INR": [down_payment],
        "Loan_Tenure_Months": [loan_tenure_months],
        "Interest_Rate": [interest_rate],
        "Monthly_Income_INR": [monthly_income]
    })
    
    # One-hot encoding karein agar categorical columns hain
    input_data = pd.get_dummies(input_data)
    
    # Columns ko model ke training columns ke sath align karein
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    # Prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Monthly EMI: **₹ {round(prediction, 2):,.2f}**")
import streamlit as st
import pickle
import numpy as np

import pickle

with open('stack_model.pkl', 'rb') as f:
    stack_model = pickle.load(f)


from tensorflow import keras
from scikeras.wrappers import KerasRegressor

def meta_mlp():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

with open('mlp_model.pkl', 'rb') as f:
    mlp_model = pickle.load(f)

with open('classifier.pkl', 'rb') as f:
    classifier_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('scaler_class.pkl', 'rb') as f:
    scaler_class = pickle.load(f)


st.title("CRS Predictor")

st.write("Enter the features to get a prediction:")

monthly_income = st.number_input("Monthly Income", value=0.0)
debt_to_income = st.number_input("Total Debt to Income Ratio", value=0.0)
credit_score = st.number_input("Credit Score", value=0.0)
net_worth = st.number_input("Net Worth", value=0.0)
loan_amount = st.number_input("Loan Amount", value=0.0)
loan_duration = st.number_input("Loan Duration (months)", value=0.0)
interest_rate = st.number_input("Interest Rate (%)", value=0.0)
monthly_payment = st.number_input("Monthly Loan Payment", value=0.0)
bankruptcy_history = st.number_input("Bankruptcy History (0 or 1)", value=0.0)
previous_defaults = st.number_input("Previous Loan Defaults (0 or 1)", value=0.0)
payment_history = st.number_input("Payment History (score)", value=0.0)
LengthOfCredit_history = st.number_input("Length of Credit History (score)", value=0.0)
total_assets = st.number_input("Total Assets", value=0.0)
total_liabilities = st.number_input("Total liabilities", value=0.0)
Checking_Account_Balance=st.number_input("Checking Account Balance", value=0.0)
Savings_Account_Balance=st.number_input("Savings Account Balance", value=0.0)
Credit_Card_utilization_rate=st.number_input("Credit Card Utilization Rate", value=0.0)

log_net_worth = np.log1p(net_worth)
log_loan_amount = np.log1p(loan_amount)
IncomeToLoanRatio = monthly_income / (monthly_payment + 1)
AssetsToLiabilities = total_assets / (total_liabilities + 1)

if st.button("Predict"):
    # Prepare the input for prediction
    input_data = np.array([[debt_to_income,credit_score, loan_duration,  interest_rate,IncomeToLoanRatio,log_net_worth,log_loan_amount, bankruptcy_history, payment_history,LengthOfCredit_history,AssetsToLiabilities,
                            monthly_payment,Credit_Card_utilization_rate,
                        previous_defaults,Checking_Account_Balance,Savings_Account_Balance]])

    scaled_input = scaler.transform(input_data)
    mlp_pred = mlp_model.predict(scaled_input)
    stack_pred = stack_model.predict(scaled_input)
    prediction = (mlp_pred + stack_pred) / 2

    st.success(f"Prediction: {prediction[0]}")

    input_class =np.array([[monthly_income,debt_to_income,credit_score,net_worth,loan_amount, loan_duration,interest_rate,monthly_payment,
    bankruptcy_history,previous_defaults,payment_history,total_assets]])

    scaled_input_class=scaler_class.transform(input_class)
    class_pred = classifier_model.predict(scaled_input_class)
    loan_status = "Yes ✅" if class_pred[0] == 1 else "No ❌"
    st.info(f"Loan Approved: {loan_status}")

#python -m streamlit run "c:\Users\adity\CRS Predictor\app.py"
#. .venv\Scripts\Activate

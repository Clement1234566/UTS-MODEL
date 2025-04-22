import streamlit as st
import pandas as pd
import pickle

# Load model dan encoder
model = pickle.load(open("xgb_model.pkl", "rb"))
gender_encoder = pickle.load(open("gender_encode.pkl", "rb"))
loan_intent_encoder = pickle.load(open("loan_intent_encode.pkl", "rb"))
education_encoder = pickle.load(open("person_education_encode.pkl", "rb"))
previous_loan_encoder = pickle.load(open("previous_loan_encode.pkl", "rb"))

# Load scaler untuk semua kolom numerik
scalers = {}
for col in ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]:
    scalers[col] = pickle.load(open(f"{col}_scaler.pkl", "rb"))

st.title("Loan Cancellation Prediction")
st.write("Masukkan detail pinjaman untuk memprediksi statusnya")

# Form Input
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Income", min_value=0, value=50000)
person_emp_length = st.number_input("Employment Length (in years)", min_value=0, value=5)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_gender = st.selectbox("Gender", ["Male", "Female"])
loan_intent = st.selectbox("Loan Intent", ["DEBT CONSOLIDATION", "EDUCATION", "HOME IMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
loan_amnt = st.number_input("Loan Amount", min_value=100, value=2000)
loan_int_rate = st.number_input("Interest Rate", min_value=0.0, max_value=50.0, value=10.0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, max_value=1.0, value=0.2)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["Yes", "No"])
cb_person_cred_hist_length = st.number_input("Credit History Length", min_value=0, value=5)

# Encode categorical
encoded_gender = gender_encoder.get(person_gender)
encoded_previous_loan = previous_loan_encoder.get(previous_loan_defaults_on_file)
encoded_education = education_encoder["person_education"].get(person_education)

# One-hot encode loan intent dan home ownership
loan_intent_ohe = loan_intent_encoder.transform([[loan_intent]]).toarray()
person_home_ownership_ohe = pickle.load(open("person_home_ownership_encoder.pkl", "rb")).transform([[person_home_ownership]]).toarray()

# Scale numeric
scaled_inputs = []
for col, val in zip(["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"],
                    [person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length]):
    scaled_inputs.append(scalers[col].transform([[val]])[0][0])

# Final input vector
final_input = pd.DataFrame([
    [encoded_gender, encoded_education, encoded_previous_loan] + scaled_inputs + list(person_home_ownership_ohe[0]) + list(loan_intent_ohe[0])
])

# Predict
if st.button("Predict Loan Status"):
    prediction = model.predict(final_input)[0]
    result = "Loan Will Be Paid" if prediction == 0 else "Loan Will Be Cancelled"
    st.success(f"Prediction: {result}")

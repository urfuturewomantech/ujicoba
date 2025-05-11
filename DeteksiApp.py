import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("pipeline_model.pkl")

# Judul Aplikasi
st.title("Deteksi Kelayakan Kredit KTA")

# Deskripsi singkat
st.write("Masukkan informasi berikut untuk memprediksi kelayakan kredit.")

# Streamlit Inputs
st.header("Loan Application Input")

person_income = st.number_input("Person Income", min_value=0, step=1)
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, step=0.01)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, step=0.01)
person_home_ownership = st.selectbox("Person Home Ownership", ["MORTGAGE", "RENT", "OTHER"])
cb_person_default_on_file = st.radio("CB Person Default on File", ["Y", "N"])

# Prediksi
if st.button("Prediksi", key="prediksi_button"):
    input_df = pd.DataFrame([{
        "person_income": person_income,
        "loan_grade": loan_grade,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "person_home_ownership": person_home_ownership,
        "cb_person_default_on_file": cb_person_default_on_file
    }])

    pred = pipeline.predict(input_df)

    if pred[0] == 1:
        st.success("Kredit LAYAK diberikan.")
    else:
        st.error("Kredit TIDAK layak diberikan.")

import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# Load pipeline
pipeline = joblib.load("pipeline_model.pkl")

# Judul Aplikasi
st.title("Deteksi Kelayakan Kredit KTA")

# Deskripsi singkat
st.write("Masukkan informasi berikut untuk memprediksi kelayakan kredit.")
loan_grade_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
}

# Streamlit Inputs
st.title("Loan Application Input")

person_income = st.number_input("Person Income", min_value=0, step=1)
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, step=0.01)
loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, step=0.01)
person_home_ownership = st.selectbox("Person Home Ownership", ["MORTGAGE", "RENT", "OTHER"])
cb_person_default_on_file = st.radio("CB Person Default on File", ["Y", "N"])

if person_home_ownership == 'MORTGAGE':
    person_home_ownership_MORTGAGE = 1
    person_home_ownership_RENT = 0
elif person_home_ownership == 'RENT':
    person_home_ownership_MORTGAGE = 0
    person_home_ownership_RENT = 1
else:
    person_home_ownership_MORTGAGE = 0
    person_home_ownership_RENT = 0


if cb_person_default_on_file == 'Y':
    cb_person_default_on_file_Y = 1
    cb_person_default_on_file_N = 0
else:
    cb_person_default_on_file_Y = 0
    cb_person_default_on_file_N = 1

# Prepare Input Data
x_data = [
    person_income,
    loan_grade_dict[loan_grade],
    loan_int_rate,
    loan_percent_income,
    person_home_ownership_MORTGAGE,
    person_home_ownership_RENT,
    cb_person_default_on_file_N,
    cb_person_default_on_file_Y
]

# st.write("Input Data: ", x_data)

# Form input
# income = st.number_input("Pendapatan (dalam USD)", min_value=0)
# age = st.number_input("Umur", min_value=18, max_value=100)
# loan_amount = st.number_input("Jumlah Pinjaman", min_value=0)
# term = st.selectbox("Jangka Waktu Pinjaman", ["short", "long"])
# credit_history = st.selectbox("Riwayat Kredit", ["good", "bad"])
# employment_status = st.selectbox("Status Pekerjaan", ["employed", "unemployed"])

# # Mapping jika perlu
# term_map = {"short": 0, "long": 1}
# history_map = {"good": 1, "bad": 0}
# employment_map = {"employed": 1, "unemployed": 0}

# Jika tombol diklik
if st.button("Prediksi"):
    # Load model
    model = joblib.load("decision_tree_model(1).pkl")  # ganti dengan nama file model kamu
    sc = joblib.load('scaler(1).pkl')
    column_names = [
    "person_income", "loan_grade", "loan_int_rate", "loan_percent_income",
    "person_home_ownership_MORTGAGE", "person_home_ownership_RENT",
    "cb_person_default_on_file_N", "cb_person_default_on_file_Y"
]

    # Buat DataFrame input
    # input_data = pd.DataFrame([{
    #     "income": income,
    #     "age": age,
    #     "loan_amount": loan_amount,
    #     "term": term_map[term],
    #     "credit_history": history_map[credit_history],
    #     "employment_status": employment_map[employment_status],
    # }])
    # x_data_array=np.array(x_data)
    # x_data_reshape=x_data_array.reshape(1,-1)
    x_data_df = pd.DataFrame([x_data], columns=column_names)
    
    # MinmMaxScaler
    # x_data_fit = sc.transform(x_data_reshape)
    x_data_scaled = sc.transform(x_data_df)
    
    # Prediksi
if st.button("Prediksi"):
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

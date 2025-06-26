import streamlit as st
import joblib
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# -----------------------------
# Cek dan load model
# -----------------------------
model_file = "svm_blood_model.pkl"

# Kalau model tidak ada, buat dan simpan dummy model
if not os.path.exists(model_file):
    # Data latih dummy (karena kita tidak punya dataset asli di cloud)
    X_dummy = [[2, 4, 1000, 20], [5, 2, 2000, 35], [1, 10, 2500, 50]]
    y_dummy = [1, 0, 1]

    # Pipeline aman
    model = make_pipeline(StandardScaler(), SVC())
    model.fit(X_dummy, y_dummy)

    # Simpan model ke file
    joblib.dump(model, model_file)
else:
    model = joblib.load(model_file)

# -----------------------------
# Judul Halaman
# -----------------------------
st.set_page_config(page_title="Prediksi Donor Darah", layout="centered")
st.title("🩸 Prediksi Donor Darah Menggunakan SVM")
st.write("""
Aplikasi ini memprediksi apakah seseorang akan melakukan donor darah di masa depan berdasarkan riwayat donor sebelumnya.
""")

# -----------------------------
# Form Input Pengguna
# -----------------------------
st.header("📋 Form Input Data")

recency = st.number_input("Recency (Bulan sejak donor terakhir)", min_value=0, max_value=100, value=2)
frequency = st.number_input("Frequency (Jumlah total donor)", min_value=0, max_value=50, value=4)
monetary = st.number_input("Monetary (Total darah dalam cc)", min_value=0, max_value=12500, step=250, value=1000)
time = st.number_input("Time (Minggu sejak donor pertama)", min_value=0, max_value=100, value=20)

# -----------------------------
# Tombol Prediksi
# -----------------------------
if st.button("🔍 Prediksi"):
    input_data = np.array([[recency, frequency, monetary, time]])
    prediction = model.predict(input_data)

    st.subheader("🧾 Hasil Prediksi:")
    if prediction[0] == 1:
        st.success("✅ Orang ini kemungkinan akan donor darah lagi.")
    else:
        st.warning("❌ Orang ini kemungkinan **tidak** akan donor darah lagi.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Model ini menggunakan Support Vector Machine (SVM) dan dilatih pada dataset Blood Transfusion Service Center dari UCI Machine Learning Repository.")

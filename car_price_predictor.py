import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os # Untuk memeriksa keberadaan file

# --- 0. Muat Data dari CSV dan Latih Model ---
# Bagian ini akan mencari dan memuat file CSV Anda.
# Pastikan file CSV ('daftar_mobil.csv') berada di folder yang sama dengan skrip Python ini.

# --- Perubahan di Sini: Menggunakan nama file daftar_mobil.csv ---
CSV_FILE_PATH = 'daftar_mobil.csv' # Nama file CSV Anda

# Periksa apakah file CSV ada
if not os.path.exists(CSV_FILE_PATH):
    st.error(f"Error: File '{CSV_FILE_PATH}' tidak ditemukan. Mohon letakkan file CSV Anda di folder yang sama dengan skrip Python ini.")
    st.stop() # Hentikan aplikasi jika file tidak ditemukan

try:
    df_train = pd.read_csv(CSV_FILE_PATH)
    # Validasi kolom yang diperlukan
    required_columns = ['Mileage', 'Year', 'Engine_Size', 'Brand', 'Price']
    if not all(col in df_train.columns for col in required_columns):
        st.error(f"Error: File CSV harus memiliki kolom berikut: {', '.join(required_columns)}")
        st.stop()
    
    # Konversi tipe data jika diperlukan (contoh: pastikan numerik adalah numerik)
    df_train['Mileage'] = pd.to_numeric(df_train['Mileage'], errors='coerce')
    df_train['Year'] = pd.to_numeric(df_train['Year'], errors='coerce')
    df_train['Engine_Size'] = pd.to_numeric(df_train['Engine_Size'], errors='coerce')
    df_train['Price'] = pd.to_numeric(df_train['Price'], errors='coerce')
    df_train.dropna(subset=required_columns, inplace=True) # Hapus baris dengan nilai NaN setelah konversi

    if df_train.empty:
        st.error("Error: Data di CSV kosong atau tidak valid setelah pembersihan. Mohon periksa isi file CSV Anda.")
        st.stop()

except Exception as e:
    st.error(f"Error saat membaca file CSV: {e}. Pastikan format CSV benar.")
    st.stop()

# Definisikan Fitur (X) dan Target (y) dari data CSV
X = df_train[['Mileage', 'Year', 'Engine_Size', 'Brand']]
y = df_train['Price']

# Definisikan fitur kategorikal dan numerik
categorical_features = ['Brand'] # 'Model' bisa ditambahkan di sini jika ingin dipakai
numerical_features = ['Mileage', 'Year', 'Engine_Size']

# Buat Preprocessor dengan ColumnTransformer untuk One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Biarkan fitur numerik apa adanya
)

# Buat Pipeline dengan Preprocessor dan Model Regresi Linear
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression()) # Model Regresi Linear
])

# Latih Model
# Pastikan ada cukup data untuk melatih model
if len(X) < 2: # Minimal 2 sampel untuk regresi linear sederhana
    st.error("Error: Tidak cukup data di CSV untuk melatih model (minimal 2 baris data yang valid).")
    st.stop()

try:
    model_pipeline.fit(X, y)
except Exception as e:
    st.error(f"Error saat melatih model regresi linear: {e}. Mohon periksa data Anda.")
    st.stop()


# --- APLIKASI STREAMLIT DIMULAI DI SINI ---

st.title("🚗 Simple Car Price Predictor (dengan Model Regresi Linear)")
st.write("Masukkan detail mobil di bawah ini untuk mendapatkan estimasi harga.")
st.write("---")

# --- Widget Input untuk Fitur Mobil ---
st.header("Masukkan Detail Mobil:")

mileage = st.slider(
    " ⏲ Odometer : Mileage (in kilometers)",
    min_value=0,
    max_value=300000,
    value=50000,
    step=1000,
    help="Jarak total yang telah ditempuh mobil."
)

year_manufacture = st.slider(
    " 📅 Tahun Pembuatan",
    min_value=1990,
    max_value=2025,
    value=2018,
    step=1,
    help="Tahun mobil diproduksi."
)

# Opsi merek diambil dari data CSV yang dimuat
brand_options_from_data = sorted(df_train['Brand'].unique().tolist())
# Tambahkan "Other" jika tidak ada di data atau jika ingin opsi default untuk merek tak dikenal
if "Other" not in brand_options_from_data:
    brand_options_from_data.append("Other")

brand = st.selectbox(
    " 🚘 Brand Mobil",
    brand_options_from_data,
    help="Pilih merek pabrikan mobil."
)

engine_size = st.slider(
    " ⛽ Ukuran Mesin (dalam Liter)",
    min_value=0.5,
    max_value=8.0,
    value=2.0,
    step=0.1,
    help="Kapasitas mesin mobil."
)

# --- Logika Prediksi Menggunakan Model yang Sudah Dilatih ---
def predict_car_price_ml(mileage, year_manufacture, brand, engine_size, model):
    input_df = pd.DataFrame([[mileage, year_manufacture, engine_size, brand]],
                            columns=['Mileage', 'Year', 'Engine_Size', 'Brand'])
    prediction = model.predict(input_df)[0]
    return int(max(prediction, 0)) # Harga tidak boleh negatif

# --- Tombol Prediksi dan Output ---
st.write("---")
if st.button("Prediksi Harga Mobil"):
    predicted_price = predict_car_price_ml(mileage, year_manufacture, brand, engine_size, model_pipeline)

    st.success(f"**Estimasi Harga Mobil (berdasarkan Model Regresi Linear):**")
    st.markdown(f"## Rp {predicted_price:,.0f}") # Format dengan koma untuk keterbacaan
    st.info(f"""
        **Penting:** Model ini dilatih menggunakan data dari file '{CSV_FILE_PATH}'. 
        Akurasi prediksi sangat bergantung pada kualitas, kuantitas, dan relevansi data di file tersebut.
        Jika data di CSV adalah data sintetis atau tidak mencerminkan pasar riil, prediksi mungkin tidak akurat.
        """)

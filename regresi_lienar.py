import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Batas maksimum kilometer yang kita asumsikan (untuk Feature Engineering)
# Sesuaikan nilai ini agar sesuai dengan jangkauan data Anda.
MAX_MILEAGE = 300000

#Gunakan nama file daftar_mobil.csv ---
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

# --- FUNGSI FEATURE ENGINEERING UNTUK MEMBALIK NILAI MILEAGE ---
def create_remaining_life(X):
    # Pastikan X diubah ke array 1D jika merupakan array 2D dengan 1 kolom
    if X.ndim == 2 and X.shape[1] == 1:
        mileage_col = X.flatten()
    else:
        mileage_col = X.flatten() # Ini adalah array 1D dari 'Mileage'
    
    # Menghitung Jarak Sisa: MAX_MILEAGE - Mileage
    remaining_life = MAX_MILEAGE - mileage_col
    
    # Mengembalikan data dalam bentuk 2D yang diharapkan oleh ColumnTransformer
    return remaining_life.reshape(-1, 1)

# Buat transformer khusus untuk Feature Engineering Mileage
mileage_transformer = Pipeline(steps=[
    # Gunakan FunctionTransformer untuk menerapkan fungsi kustom
    ('remaining_life', FunctionTransformer(create_remaining_life, validate=False))
])
# ------------------------------------------------------------------

# Definisikan fitur kategorikal dan numerik
categorical_features = ['Brand']
numerical_features = ['Mileage', 'Year', 'Engine_Size'] # Tetap gunakan nama kolom asli

# Buat Preprocessor dengan ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        # Terapkan transformasi Mileage (Jarak Sisa) hanya pada kolom 'Mileage'
        ('mileage_rev', mileage_transformer, ['Mileage']),
        # Terapkan One-Hot Encoding pada fitur Brand
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        # Biarkan fitur numerik lainnya apa adanya
        ('num_passthrough', 'passthrough', ['Year', 'Engine_Size'])
    ],
    remainder='drop' # Pastikan hanya fitur yang ditransformasi yang digunakan
)

# Buat Pipeline dengan Preprocessor dan Model Regresi Linear
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression()) # Model Regresi Linear tetap dipertahankan
])

# Latih Model
if len(X) < 2:
    st.error("Error: Tidak cukup data di CSV untuk melatih model (minimal 2 baris data yang valid).")
    st.stop()

try:
    model_pipeline.fit(X, y)
except Exception as e:
    # Error handling di sini akan menangkap error jika terjadi masalah lain saat pelatihan
    st.error(f"Error saat melatih model regresi linear: {e}. Mohon periksa data Anda.")
    st.stop()


# --- APLIKASI STREAMLIT DIMULAI DI SINI ---

st.title("🚗 Simple Second Car Price Predictor (Regresi Linear dengan Jarak Sisa)")
st.write(f"Model ini menggunakan Regresi Linear dan dijamin bahwa **prediksi harga akan menurun seiring dengan kenaikan Mileage** karena kami menggunakan fitur 'Jarak Sisa' (dihitung dari {MAX_MILEAGE:,} km).")
st.write("---")

# --- Tampilkan Tabel Data di Sini ---
st.header("Data-data mobil second")
st.write("Berikut adalah data lengkap mobil second yang sering dicari:")
st.dataframe(df_train)
st.write("---")

# --- Widget Input untuk Fitur Mobil ---
st.header("Masukkan Detail Mobil:")

# MIN dan MAX slider disesuaikan dengan asumsi MAX_MILEAGE
mileage = st.slider(
    " ⏲ Odometer : Mileage (in kilometers)",
    min_value=0,
    max_value=MAX_MILEAGE,
    value=50000,
    step=1000,
    help="Jarak total yang telah ditempuh mobil. Nilai ini akan dibalik untuk model."
)

year_manufacture = st.slider(
    " 📅 Tahun Pembuatan",
    min_value=2015,
    max_value=2025,
    value=2018,
    step=1,
    help="Tahun mobil diproduksi."
)

# Opsi merek diambil dari data CSV yang dimuat
brand_options_from_data = sorted(df_train['Brand'].unique().tolist())
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
    # Data input tetap dalam format asli
    input_df = pd.DataFrame([[mileage, year_manufacture, engine_size, brand]],
                            columns=['Mileage', 'Year', 'Engine_Size', 'Brand'])
    
    # Pipeline akan secara otomatis menerapkan transformasi 'Jarak Sisa' sebelum prediksi
    prediction = model.predict(input_df)[0]
    return int(max(prediction, 0)) # Harga tidak boleh negatif

# --- Tombol Prediksi dan Output ---
st.write("---")
if st.button("Prediksi Harga Mobil"):
    predicted_price = predict_car_price_ml(mileage, year_manufacture, brand, engine_size, model_pipeline)

    st.success(f"**Estimasi Harga Mobil (berdasarkan Model Regresi Linear dengan Jarak Sisa):**")
    st.markdown(f"## Rp {predicted_price:,.0f}") # Format dengan koma untuk keterbacaan
    st.info(f"""
        **Penting:** Karena fitur Mileage telah direkayasa menjadi Jarak Sisa, hubungan negatif Mileage terhadap Harga kini terjamin. Namun, model Regresi Linear tetap merupakan model yang sederhana dan mungkin kurang akurat dibandingkan model yang lebih canggih.
        """)

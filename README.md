# Praktikum_DataMining

# Cara Menjalankan Aplikasi Prediksi Harga Mobil Ini

Aplikasi ini adalah prediktor harga mobil sederhana yang dibangun menggunakan Streamlit dan model Regresi Linear. Data pelatihan dimuat dari file CSV.

## Persiapan Awal

Pastikan Anda memiliki hal-hal berikut:

1.  **Python Terinstal:** Pastikan Anda memiliki Python 3.7 atau lebih tinggi terinstal di sistem Anda.
2.  **File Aplikasi Python:** Simpan kode aplikasi Python Anda (yang telah kita kerjakan sebelumnya) ke dalam file bernama `car_price_predictor_from_csv.py`.
3.  **File Data CSV:** Pastikan Anda memiliki file data CSV dengan nama `daftar_mobil.csv` di folder yang **sama** dengan skrip Python Anda.

### Struktur File CSV (`daftar_mobil.csv`)

File CSV Anda **harus** memiliki kolom-kolom berikut:
`Brand`, `Model` (opsional, tidak digunakan model), `Year`, `Mileage`, `Engine_Size`, `Price`.

**Penting:** Pastikan nilai pada kolom `Engine_Size` di file CSV Anda dalam satuan **Cubic Centimeter (cc)**. Jika data asli Anda dalam Liter, Anda harus mengonversinya ke cc terlebih dahulu (kalikan nilai Liter dengan 1000).

Contoh isi `daftar_mobil.csv`:
```csv
Brand,Model,Year,Mileage,Engine_Size,Price
Toyota,Avanza,2023,5000,1500,260000000
Toyota,Innova Zenix,2024,1000,2000,470000000
Honda,Brio Satya,2023,8000,1200,180000000
Hyundai,Ioniq 5,2023,8000,0,780000000
# ... data lainnya
 

import streamlit as st
import pandas as pd
import joblib
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Anemia (Random Forest)",
    page_icon="🩸",
    layout="centered"
)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    """Hanya memuat model Random Forest Robust."""
    model_path = 'models/rf_robust_model.pkl'
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ File model tidak ditemukan di: {model_path}")
        st.warning("Pastikan Anda sudah menyimpan model ke folder 'models/'.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Load model saat aplikasi dimulai
model = load_model()

# --- JUDUL & DESKRIPSI ---
st.title("🩸 Sistem Deteksi Anemia")
st.markdown("""
Aplikasi ini menggunakan algoritma **Random Forest** untuk mendeteksi risiko anemia.
Model ini dirancang khusus untuk bekerja optimal meskipun hanya menggunakan parameter sel darah mikro.
""")
st.divider()

# --- SIDEBAR: INPUT DATA ---
st.sidebar.header("📝 Masukkan Data Pasien")

def user_input_features():
    # Gender (Diperlukan untuk data entry, meski nanti dihapus oleh model ini)
    gender_label = st.sidebar.selectbox("Jenis Kelamin", ("Wanita", "Pria"))
    gender = 0 if gender_label == "Wanita" else 1
    
    # Input Numerik
    hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.0, step=0.1)
    mch = st.sidebar.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=22.0, step=0.1)
    mchc = st.sidebar.number_input("MCHC (g/dL)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
    mcv = st.sidebar.number_input("MCV (fL)", min_value=0.0, max_value=150.0, value=85.0, step=0.1)
    
    data = {
        'Gender': gender,
        'Hemoglobin': hemoglobin,
        'MCH': mch,
        'MCHC': mchc,
        'MCV': mcv
    }
    return pd.DataFrame([data])

# Ambil input dari user
input_df = user_input_features()

# Tampilkan input di halaman utama
st.subheader("Data Pasien:")
st.dataframe(input_df)

# --- TOMBOL PREDIKSI ---
if st.button("🔍 Analisis Sekarang"):
    
    if model is None:
        st.stop()

    # --- 1. FEATURE ENGINEERING ---
    # Langkah ini WAJIB dilakukan sebelum data masuk ke model
    process_df = input_df.copy()
    
    # Hitung Mean Red Cell Features
    process_df['Mean_RCF'] = (process_df['MCHC'] + process_df['MCV'] + process_df['MCH']) / 3
    
    # Hitung Rasio Hb/MCH (Menghindari pembagian nol)
    process_df['Hb_MCH_Ratio'] = process_df.apply(
        lambda x: x['Hemoglobin'] / x['MCH'] if x['MCH'] != 0 else 0, axis=1
    )

    # --- 2. PERSIAPAN KHUSUS RANDOM FOREST ROBUST ---
    # Model 'rf_robust' dilatih TANPA fitur Gender dan Hemoglobin.
    # Kita harus membuang kolom tersebut agar sesuai dengan input yang diharapkan model.
    final_features = process_df.drop(columns=['Gender', 'Hemoglobin'])

    # --- 3. PREDIKSI ---
    prediction = model.predict(final_features)[0]
    prob = model.predict_proba(final_features)[:, 1][0]

    # --- 4. TAMPILKAN HASIL ---
    st.divider()
    st.subheader("Hasil Diagnosa:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("🔴 POSITIF ANEMIA")
            st.write("Berdasarkan analisis fitur sel darah, pasien terindikasi anemia.")
        else:
            st.success("🟢 NEGATIF (NORMAL)")
            st.write("Profil sel darah pasien dalam batas normal.")
            
    with col2:
        st.metric(label="Tingkat Keyakinan AI", value=f"{prob:.1%}")
        st.progress(prob)

    # Disclaimer
    st.info("⚠️ Catatan: Hasil ini menggunakan model Random Forest Robust yang berfokus pada indeks sel darah (MCH, MCV, MCHC).")
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# ==========================================
# 1. KONFIGURASI & SETUP HALAMAN
# ==========================================
st.set_page_config(
    page_title="Hoax Buster AI",
    page_icon="🛡️",
    layout="wide", # Layout wide agar lebih lega
    initial_sidebar_state="expanded"
)

# Inisialisasi Session State untuk History
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ==========================================
# 2. LOAD MODEL & VECTORIZER
# ==========================================
@st.cache_resource
def load_models():
    # Load Vectorizer
    vect = joblib.load("vectorizer.pkl")
    # Load Model Klasifikasi
    model = joblib.load("model_xgboost_under.pkl")
    return vect, model

# Sidebar - Bagian Header & Status Model
with st.sidebar:
    st.title("🛡️ Hoax Buster AI")
    st.markdown("---")
    st.write("Sistem cerdas untuk memverifikasi kebenaran berita menggunakan **XGBoost**.")
    
    # Indikator Status Model
    try:
        vectorizer, xgb_model = load_models()
        st.success("Status: Model AI Aktif", icon="✅")
    except FileNotFoundError:
        st.error("File model tidak ditemukan!", icon="❌")
        st.stop()

# ==========================================
# 3. FUNGSI TAMBAHAN
# ==========================================
def simpan_riwayat(teks, label, conf):
    # Simpan waktu saat ini
    waktu = datetime.now().strftime("%H:%M:%S")
    
    # Buat dictionary data baru
    data_baru = {
        "Waktu": waktu,
        "Teks (Cuplikan)": teks[:50] + "..." if len(teks) > 50 else teks,
        "Prediksi": label,
        "Confidence": f"{conf:.2f}%"
    }
    
    # Masukkan ke list history (paling atas)
    st.session_state['history'].insert(0, data_baru)

# ==========================================
# 4. HALAMAN UTAMA (MAIN UI)
# ==========================================
st.header("🔍 Analisis Kebenaran Berita")
st.markdown("Tempelkan teks berita yang ingin Anda verifikasi di bawah ini.")

# Input Area
input_text = st.text_area("Isi Berita:", height=200, placeholder="Contoh: Pemerintah akan membagikan bantuan sosial sebesar 5 juta rupiah mulai besok...")

col_btn, col_space = st.columns([1, 5])
with col_btn:
    tombol_analisis = st.button("🚀 Analisis Sekarang", type="primary", use_container_width=True)

# Logika Prediksi
if tombol_analisis:
    if input_text.strip():
        try:
            with st.spinner("Sedang memproses teks..."):
                # 1. Preprocessing
                text_vectorized = vectorizer.transform([input_text])
                
                # 2. Prediksi
                prediksi = xgb_model.predict(text_vectorized)[0]
                proba = xgb_model.predict_proba(text_vectorized)
                
                # 3. Mapping
                labels = ["Fakta", "Hoax"] # Pastikan urutan ini sesuai training Anda (0/1)
                hasil_label = labels[prediksi]
                confidence = proba[0][prediksi] * 100
                
                # 4. Simpan ke History
                simpan_riwayat(input_text, hasil_label, confidence)
                st.toast("Analisis selesai! Disimpan ke Riwayat.", icon="💾")

            # 5. Tampilkan Hasil dengan UI Menarik
            st.markdown("---")
            st.subheader("📊 Hasil Analisis")
            
            # Layout Hasil menggunakan Columns
            res_col1, res_col2, res_col3 = st.columns([2, 1, 1])
            
            with res_col1:
                if hasil_label == "Hoax":
                    st.error(f"### 🚨 Terdeteksi: {hasil_label}")
                    st.markdown("Berita ini memiliki indikasi kuat sebagai informasi palsu. Harap verifikasi sumber resminya.")
                else:
                    st.success(f"### ✅ Terdeteksi: {hasil_label}")
                    st.markdown("Berita ini terindikasi sebagai informasi yang valid/fakta.")
            
            with res_col2:
                st.metric("Confidence Score", f"{confidence:.1f}%")
            
            # with res_col3:
            #     # Progress bar untuk visualisasi probabilitas HOAX
            #     prob_hoax = proba[0][1] # Ambil probabilitas kelas 1 (Hoax)
            #     st.write("**Skor Hoax:**")
            #     st.progress(prob_hoax, text=f"{prob_hoax*100:.1f}%")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Mohon isi teks berita terlebih dahulu.")

# ==========================================
# 5. SIDEBAR RIWAYAT (HISTORY)
# ==========================================
with st.sidebar:
    st.markdown("---")
    st.subheader("📜 Riwayat Analisis")
    
    if st.session_state['history']:
        # Tombol hapus history
        if st.button("Hapus Riwayat"):
            st.session_state['history'] = []
            st.rerun()
            
        # Konversi list ke DataFrame agar tampil rapi sebagai tabel
        df_history = pd.DataFrame(st.session_state['history'])
        
        # Tampilkan tabel tanpa index
        st.dataframe(df_history, hide_index=True, use_container_width=True)
    else:
        st.info("Belum ada riwayat analisis.")
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# ==========================================
# 1. KONFIGURASI & SETUP HALAMAN
# ==========================================
st.set_page_config(
    page_title="Cek Politik AI",
    page_icon="🛡️",
    layout="wide",
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
    vect = joblib.load("tfidf_vectorizer.pkl")
    # Load Model Klasifikasi
    model = joblib.load("model_xgboost_under.pkl")
    return vect, model

# Sidebar - Bagian Header & Status Model
with st.sidebar:
    st.title("🛡️ Cek Politik AI")
    st.markdown("---")
    st.write("Sistem cerdas untuk memverifikasi kebenaran berita menggunakan Machine Learning.")
    
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
st.header("Analisis Kebenaran Berita Politik Indonesia")
st.markdown("Tempelkan teks berita yang ingin Anda verifikasi di bawah ini.")

# Form Section
with st.form(key="analisis_form"):
    # Input Area
    input_text = st.text_area(
        "Isi Berita:", 
        height=200, 
        placeholder="Contoh: Pemerintah akan membagikan bantuan sosial sebesar 5 juta rupiah mulai besok..."
    )
    
    # button submit
    submit_button = st.form_submit_button(label="🚀 Analisis Sekarang", type="primary")

# Logika Prediksi
if submit_button:
    # cek apakah kosong
    if not input_text.strip():
        st.warning("Mohon isi teks berita terlebih dahulu.")
    else:
        # hitung jumlah kata
        word_count = len(input_text.strip().split())
        
        # validasi minimal 100 kata
        if word_count < 100:
            st.error(f"❌ Teks harus minimal 100 kata. Saat ini Anda memiliki {word_count} kata.")
            st.info(f"Silahkan tambahkan **{100 - word_count}** kata lagi agar analisis lebih akurat.")
        else:
            # proses analisis
            try:
                with st.spinner("Sedang memproses teks..."):
                    # 1. Preprocessing
                    text_vectorized = vectorizer.transform([input_text])
                    
                    # 2. Prediksi
                    prediksi = xgb_model.predict(text_vectorized)[0]
                    proba = xgb_model.predict_proba(text_vectorized)
                    
                    # 3. Mapping
                    labels = ["Fakta", "Hoax"]
                    hasil_label = labels[prediksi]
                    confidence = proba[0][prediksi] * 100
                    
                    # 4. Simpan ke History
                    simpan_riwayat(input_text, hasil_label, confidence)
                    st.toast("Analisis selesai! Disimpan ke Riwayat.", icon="💾")

                # 5. Tampilkan Hasil
                st.markdown("---")
                st.subheader("📊 Hasil Analisis")
                
                # Layout Hasil menggunakan Columns
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    if hasil_label == "Hoax":
                        st.error(f"### 🚨 Terdeteksi: {hasil_label}")
                        st.markdown("Berita ini memiliki indikasi kuat sebagai informasi palsu. Harap verifikasi sumber resminya.")
                    else:
                        st.success(f"### ✅ Terdeteksi: {hasil_label}")
                        st.markdown("Berita ini terindikasi sebagai informasi yang valid/fakta.")
                
                with res_col2:
                    st.metric("Confidence Score", f"{confidence:.1f}%")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

# ==========================================
# 5. SIDEBAR HISTORY
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
        
# troubleshoot
# st.write(vectorizer)

# st.write("Punya idf_?", hasattr(vectorizer, "idf_"))
# if hasattr(vectorizer, "idf_"):
#     st.write("Panjang idf_:", len(vectorizer.idf_))
# else:
#     st.write("idf_ TIDAK ADA (belum fit)")

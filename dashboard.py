import streamlit as st
from PIL import Image

# ======== STYLE ========
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(180deg, #ffd1dc, #ffb6c1, #ff8fab);
    font-family: 'Poppins', sans-serif;
}

/* Sidebar style */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffdae0, #ffc0cb, #ff9aae);
    color: #5c0036;
    border-right: 3px solid #ff80a6;
    box-shadow: 4px 0 20px rgba(255, 100, 150, 0.3);
}

/* Header in sidebar */
.sidebar-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #b3005a;
    text-shadow: 1px 1px 3px #ffafc0;
    margin-bottom: 0.5rem;
}

/* Deskripsi box */
.desc-box {
    background-color: rgba(255, 240, 245, 0.7);
    border: 2px solid #ff8fab;
    border-radius: 12px;
    padding: 10px;
    margin-top: 10px;
    box-shadow: inset 0 0 10px rgba(255, 150, 180, 0.4);
}

/* Cute emoji header */
h1 {
    color: #b3005a;
    text-shadow: 1px 1px 4px #ffc0cb;
}
</style>
""", unsafe_allow_html=True)

# ======== SIDEBAR ========
st.sidebar.markdown('<div class="sidebar-title">🌸 Pilih Mode</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# Sidebar dynamic description
if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div class="desc-box">
    🔍 <b>Deteksi Objek (YOLO)</b><br>
    Gunakan model YOLO (.pt) untuk mendeteksi karakter seperti <b>Spongebob</b> 🧽 dan <b>Patrick</b> 🌟.<br>
    Upload gambar → model akan menandai objek secara otomatis!
    </div>
    """, unsafe_allow_html=True)
elif mode == "Klasifikasi Gambar":
    st.sidebar.markdown("""
    <div class="desc-box">
    🏡 <b>Klasifikasi Gambar</b><br>
    Model Keras (.h5) membedakan gambar <b>Indoor 🪴</b> dan <b>Outdoor 🌤️</b>.<br>
    Cocok untuk dataset pemandangan atau ruangan!
    </div>
    """, unsafe_allow_html=True)

# ======== MAIN AREA ========
st.markdown("""
<h1>💗 PinkVision: Cute Image & Object Detector 💗</h1>
<p style='color:#7b0046;'>Selamat datang di <b>PinkVision</b> 💖 — aplikasi pendeteksi dan pengklasifikasi gambar berbasis AI dengan tema pink aesthetic!</p>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("📸 Unggah gambar kamu di sini", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success("✨ Gambar berhasil diunggah!")
    for file in uploaded_files:
        img = Image.open(file)
        st.image(img, caption=file.name, use_column_width=True)
        st.markdown(f"<p style='color:#ff4f9a;'>Nama file: {file.name}</p>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b3005a;'>Made with 💕 by <b>Emmy Nora</b> 🌷</p>", unsafe_allow_html=True)

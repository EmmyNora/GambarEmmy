import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ======== STYLE ========
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
    overflow: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe3eb, #ffc6d5, #ff9ec4);
    color: #4a0032;
    border-right: 3px solid #ff82a9;
    box-shadow: 4px 0 15px rgba(255, 100, 150, 0.3);
    padding-top: 1rem;
}

.sidebar-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #b3005a;
    text-shadow: 1px 1px 3px #ffc0cb;
    margin-bottom: 1rem;
}

.desc-box {
    background-color: rgba(255, 240, 245, 0.7);
    border: 2px solid #ff8fab;
    border-radius: 12px;
    padding: 10px;
    margin-top: 10px;
    box-shadow: inset 0 0 10px rgba(255, 150, 180, 0.4);
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 2.3rem;
    color: #b3005a;
    font-weight: 800;
    text-shadow: 2px 2px 6px #ffbad5;
    margin-top: 2rem;
}

/* Upload box */
.upload-box {
    border: 3px dashed #ff8fab;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    background-color: rgba(255, 255, 255, 0.5);
    margin-top: 20px;
    box-shadow: 0px 0px 15px rgba(255, 150, 180, 0.3);
}

/* ========== Animasi Mengambang ========== */
@keyframes floaty {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-15px);}
    100% {transform: translateY(0px);}
}

.bg-item {
    position: absolute;
    opacity: 0.85;
    z-index: -1;
    animation: floaty 6s ease-in-out infinite;
}

/* SpongeBob & Patrick (ikon kartun lucu) */
.spongebob {
    width: 120px;
    bottom: 40px;
    left: 10%;
    animation-delay: 0s;
}
.patrick {
    width: 100px;
    bottom: 40px;
    right: 10%;
    animation-delay: 1.5s;
}

/* Indoor plant & outdoor sun */
.plant {
    width: 90px;
    top: 80px;
    left: 20%;
    animation-delay: 2s;
}
.sun {
    width: 100px;
    top: 40px;
    right: 15%;
    animation-delay: 3s;
}
</style>
""", unsafe_allow_html=True)

# ======== SIDEBAR ========
st.sidebar.markdown('<div class="sidebar-title">🌸 Pilih Mode</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div class="desc-box">
    🔍 <b>Deteksi Objek (YOLO)</b><br>
    Gunakan model YOLO (.pt) untuk mengenali karakter seperti
    <b>Spongebob</b> 🧽 dan <b>Patrick</b> 🌟 pada gambar yang kamu unggah!
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="desc-box">
    🏡 <b>Klasifikasi Gambar</b><br>
    Gunakan model Keras (.h5) untuk membedakan gambar
    <b>Indoor 🪴</b> dan <b>Outdoor 🌤️</b> secara otomatis!
    </div>
    """, unsafe_allow_html=True)

# ======== MAIN LAYOUT ========
st.markdown('<div class="main-title">💗 PinkVision: Cute Image & Object Detector 💗</div>', unsafe_allow_html=True)

# ======== Animasi Background ========
st.markdown(f"""
<img src="data:image/png;base64,{open('/mnt/data/1cf0c3f3-4b91-4755-97f5-1df7bc2f3a60.png','rb').read().encode('base64').decode()}" class="bg-item spongebob">
<img src="https://i.ibb.co/6bMmyxF/patrick.png" class="bg-item patrick">
<img src="https://i.ibb.co/WyNYn4k/plant.png" class="bg-item plant">
<img src="https://i.ibb.co/mT0kZwh/sun.png" class="bg-item sun">
""", unsafe_allow_html=True)

# ======== Upload Area ========
st.markdown('<div class="upload-box">📸 <b>Seret dan lepas (drag & drop)</b> gambar kamu di sini 💕</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    st.success("✨ Gambar berhasil diunggah!")
    for file in uploaded_files:
        img = Image.open(file)
        st.image(img, caption=file.name, use_column_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b3005a;'>Made with 💕 by <b>Emmy Nora</b> 🌷</p>", unsafe_allow_html=True)

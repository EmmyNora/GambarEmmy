import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="💗 PinkVision", layout="wide")

# -----------------------------
# STYLE
# -----------------------------
st.markdown(
    """
    <style>
    /* 🌸 Background gradasi pink lembut */
    .stApp {
        background: linear-gradient(180deg, #ffe6f0 0%, #ffb6d1 50%, #ff9ecb 100%);
        font-family: 'Poppins', sans-serif;
        overflow-x: hidden;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 230, 240, 0.8);
        color: #4a0032;
        border-right: 2px solid rgba(255,150,180,0.25);
        box-shadow: 4px 0 18px rgba(255,100,150,0.1);
    }

    /* Judul utama */
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        color: #b3005a;
        font-weight: 800;
        text-shadow: 2px 2px 10px rgba(255,180,210,0.6);
        margin-top: 1.8rem;
        margin-bottom: 0.5rem;
    }

    /* Kotak upload gambar — diperkecil */
    .upload-box {
        border: 3px dashed rgba(255,140,170,0.7);
        border-radius: 14px;
        padding: 20px;
        width: 60%;
        margin: 20px auto;
        text-align: center;
        background-color: rgba(255,255,255,0.5);
        box-shadow: 0 4px 18px rgba(255,150,180,0.1);
    }

    /* Gambar SpongeBob kanan bawah */
    .stApp::after {
        content: "";
        position: fixed;
        bottom: 0;
        right: 0;
        width: 320px;
        height: 320px;
        background-image: url('https://i.pinimg.com/736x/a1/aa/58/a1aa5870adbb34ef6e20b9e9d6c8deb6.jpg');
        background-size: contain;
        background-repeat: no-repeat;
        opacity: 0.25;
        z-index: 0;
    }

    .block-container {
        position: relative;
        z-index: 10;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #b3005a;
        font-weight: 600;
        margin-top: 3rem;
    }

    @media (max-width: 600px) {
        .main-title { font-size: 1.6rem; }
        .upload-box { width: 85%; padding: 16px; }
        .stApp::after { width: 180px; height: 180px; opacity: 0.3; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# SIDEBAR: Pilih Mode
# -----------------------------
st.sidebar.markdown(
    '<div style="font-weight:800; color:#b3005a; font-size:18px;">🌷 Pilih Mode</div>',
    unsafe_allow_html=True
)

mode = st.sidebar.radio("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div style="background: rgba(255,240,245,0.8); padding:10px; border-radius:10px; border:1px solid rgba(255,140,170,0.4);">
    🔍 <b>Deteksi Objek (YOLO)</b><br>
    Gunakan model YOLO (.pt) untuk mengenali karakter Spongebob 🧽 & Patrick 🌟
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background: rgba(255,240,245,0.8); padding:10px; border-radius:10px; border:1px solid rgba(255,140,170,0.4);">
    🏡 <b>Klasifikasi Gambar</b><br>
    Gunakan model Keras (.h5) untuk membedakan Indoor 🪴 dan Outdoor 🌤️
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# MAIN CONTENT
# -----------------------------
st.markdown('<div class="main-title">💗 PinkVision: Cute Image & Object Detector 💗</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-box">📸 <b>Seret dan lepas (drag & drop)</b> gambar kamu di sini 💕</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success("✨ Gambar berhasil diunggah!")
    for file in uploaded_files:
        img = Image.open(file)
        st.image(img, caption=file.name, use_column_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<div class='footer'>Made with 💕 by <b>Emmy Nora</b> 🌷</div>", unsafe_allow_html=True)

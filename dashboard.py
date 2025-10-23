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
# HELPERS
# -----------------------------
def image_to_data_uri(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """Encode image bytes to data URI for embedding in CSS."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

# -----------------------------
# UI: Background selection (optional)
# -----------------------------
st.sidebar.markdown("## 🎨 Background (opsional)")
bg_file = st.sidebar.file_uploader(
    "Upload background image (.png/.jpg) — kalau tidak, akan pakai default",
    type=["png", "jpg", "jpeg"]
)

# Default image (SpongeBob & Patrick)
DEFAULT_BG_URL = "https://i.pinimg.com/736x/a1/aa/58/a1aa5870adbb34ef6e20b9e9d6c8deb6.jpg"

# Prepare background source: either uploaded file -> data URI, or remote URL
if bg_file:
    raw = bg_file.read()
    mime = "image/png" if bg_file.type == "image/png" else "image/jpeg"
    bg_data_uri = image_to_data_uri(raw, mime)
    bg_source = bg_data_uri
else:
    bg_source = DEFAULT_BG_URL

# -----------------------------
# STYLE: Gradient overlay + SpongeBob kecil di kanan bawah
# -----------------------------
st.markdown(
    f"""
    <style>
    /* 🌸 Background utama dengan SpongeBob kecil di kanan bawah */
    .stApp {{
        background-color: #ffe6f0;
        background-image:
            linear-gradient(180deg, rgba(255,223,230,0.88), rgba(255,185,200,0.60)),
            url("{bg_source}");
        background-size: 300px auto;       /* kecilin SpongeBob */
        background-position: right 30px bottom 30px;  /* pojok kanan bawah */
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
        overflow-x: hidden;
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(255,236,242,0.85), rgba(255,220,235,0.75));
        color: #4a0032;
        border-right: 2px solid rgba(255,130,169,0.25);
        box-shadow: 4px 0 18px rgba(255,100,150,0.08);
        padding-top: 1rem;
    }}

    /* Judul utama */
    .main-title {{
        text-align: center;
        font-size: 2.4rem;
        color: #b3005a;
        font-weight: 800;
        text-shadow: 2px 2px 10px rgba(255,190,210,0.6);
        margin-top: 1.8rem;
    }}

    /* Box upload gambar */
    .upload-box {{
        border: 3px dashed rgba(255,140,170,0.7);
        border-radius: 16px;
        padding: 34px;
        text-align: center;
        background-color: rgba(255,255,255,0.60);
        margin-top: 22px;
        box-shadow: 0 6px 28px rgba(255,150,180,0.12);
    }}

    .block-container {{
        position: relative;
        z-index: 10;
    }}

    @media (max-width: 600px) {{
        .main-title {{ font-size: 1.6rem; }}
        .upload-box {{ padding: 20px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# PAGE CONTENT
# -----------------------------
st.sidebar.markdown(
    '<div style="font-weight:800; color:#b3005a; font-size:18px;">🌸 Pilih Mode</div>',
    unsafe_allow_html=True
)
mode = st.sidebar.radio("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div style="background: rgba(255,240,245,0.7); padding:10px; border-radius:10px; border:1px solid rgba(255,140,170,0.4);">
    🔍 <b>Deteksi Objek (YOLO)</b><br>
    Gunakan model YOLO (.pt) untuk mengenali karakter Spongebob 🧽 & Patrick 🌟
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background: rgba(255,240,245,0.7); padding:10px; border-radius:10px; border:1px solid rgba(255,140,170,0.4);">
    🏡 <b>Klasifikasi Gambar</b><br>
    Gunakan model Keras (.h5) untuk membedakan Indoor 🪴 dan Outdoor 🌤️
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">💗 PinkVision: Cute Image & Object Detector 💗</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-box">📸 <b>Seret dan lepas (drag & drop)</b> gambar kamu di sini 💕</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success("✨ Gambar berhasil diunggah!")
    for file in uploaded_files:
        img = Image.open(file)
        st.image(img, caption=file.name, use_column_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b3005a;'>Made with 💕 by <b>Emmy Nora</b> 🌷</p>", unsafe_allow_html=True)

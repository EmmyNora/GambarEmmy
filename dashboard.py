import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="💖 PinkVision: Smart & Cute AI 💖",
    page_icon="💖",
    layout="wide"
)

# ==========================
# STYLING
# ==========================
st.markdown("""
    <style>
        body {
            background-color: pink;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #ffb6c1;
        }
        [data-testid="stSidebar"] {
            background-color: #ffc0cb;
        }
        [data-testid="stHeader"] {
            background-color: #ffb6c1;
        }
        div.stButton > button {
            background-color: #ff69b4;
            color: white;
            border-radius: 10px;
            border: none;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #ff85c1;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.markdown('<h2 style="text-align:center;">💖 PinkVision Menu 💖</h2>', unsafe_allow_html=True)
st.sidebar.write("Gunakan aplikasi ini untuk mendeteksi objek dengan model YOLO!")

# Upload file gambar
uploaded_files = st.sidebar.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ==========================
# MODEL YOLO
# ==========================
model_path = "best.pt"  # pastikan file model YOLO kamu tersedia di folder yang sama
model = YOLO(model_path)

# ==========================
# DETEKSI OBJEK
# ==========================
detection_results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption=uploaded_file.name, use_column_width=True)

        # Jalankan deteksi YOLO
        results = model.predict(source=np.array(image_pil), save=False)
        result_img = results[0].plot()  # visualisasi hasil deteksi

        # Simpan hasil
        detection_results.append(result_img)

# ==========================
# HASIL DETEKSI OBJEK (DIPERBAIKI)
# ==========================
if detection_results:
    st.markdown("💖 <b>Hasil Deteksi Objek</b> 💖", unsafe_allow_html=True)

    if len(uploaded_files) == 1:
        # Jika hanya 1 gambar, tampilkan besar penuh (1 layar)
        st.image(detection_results[0], use_column_width=True)
    else:
        # Jika lebih dari 1 gambar, tampilkan 2 gambar per baris
        cols = st.columns(2)
        for i, result in enumerate(detection_results):
            with cols[i % 2]:
                st.image(result, use_column_width=True)

    st.success("✅ Objek berhasil terdeteksi!")

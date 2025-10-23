import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import torch
import io
from torchvision import models, transforms

# --- Page setup ---
st.set_page_config(page_title="PinkVision 💖", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #ffe6f2, #ffcce0);
    }
    .stApp {
        color: #5c0a3f;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border: 3px dashed #ff80b3;
        background-color: #ffe6f2;
        border-radius: 15px;
        padding: 30px;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #ff4da6;
    }
    h1 {
        text-align: center;
        color: #d63384;
        font-weight: 800;
        text-shadow: 1px 1px 2px white;
    }
    .pink-box {
        background-color: #ffe6f2;
        border: 2px solid #ff99cc;
        border-radius: 10px;
        padding: 10px 15px;
        margin-top: 15px;
        color: #cc0066;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>💖 PinkVision: Cute Image & Object Detector 💖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>📸 Seret dan lepas (drag & drop) gambar kamu di sini 💕</p>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("### 🌷 Pilih Mode")
mode = st.sidebar.radio("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div class='pink-box'>
    🔍 <b>Deteksi Objek (YOLO)</b><br>
    Gunakan model YOLO (.pt) untuk mengenali karakter<br>
    🧽 Spongebob & 🩷 Patrick ☀️
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class='pink-box'>
    🖼️ <b>Klasifikasi Gambar</b><br>
    Gunakan model CNN (.h5) untuk klasifikasi gambar lucu 💕
    </div>
    """, unsafe_allow_html=True)

# --- Upload file section ---
col1, col2 = st.columns([1.1, 1])  # kiri lebih besar
with col1:
    uploaded_file = st.file_uploader("Drag and drop files here", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Asli 💕", use_container_width=True)

with col2:
    st.write("### 📊 Hasil Deteksi / Klasifikasi")
    if uploaded_file is not None:
        if mode == "Deteksi Objek (YOLO)":
            # Contoh: load YOLO model
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt', force_reload=False)
            results = model(image)
            results.render()
            detected_img = Image.fromarray(results.ims[0])
            st.image(detected_img, caption="Hasil Deteksi 🩷", use_container_width=True)
        else:
            # Contoh dummy CNN (bisa ganti model sendiri)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            img_tensor = transform(image).unsqueeze(0)
            model = models.resnet18(pretrained=True)
            model.eval()
            with torch.no_grad():
                outputs = model(img_tensor)
            st.success("✨ Klasifikasi berhasil! (contoh output dummy)")

# --- Footer ---
st.markdown("<p style='text-align:center;'>Made with 💕 by Emmy Nora 🌷</p>", unsafe_allow_html=True)

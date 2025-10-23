import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# 🌸 KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="🌸 PinkVision: Smart & Cute AI 🌸",
    page_icon="💖",
    layout="wide"
)

# ==========================
# 🌷 CUSTOM CSS (TEMA PINK)
# ==========================
st.markdown("""
    <style>
    /* ======== BASE STYLE ======== */
    .stApp {
        background: linear-gradient(180deg, #ffe9f0, #ffd6e7);
        color: #6a004f;
        font-family: "Poppins", sans-serif;
    }

    /* ======== HEADER ======== */
    h1 {
        text-align: center;
        color: #8c005c;
        font-weight: 800;
        font-size: 2.4em;
        letter-spacing: 1px;
        text-shadow: 2px 2px 5px rgba(255, 182, 193, 0.6);
        margin-top: 10px;
    }

    /* ======== SIDEBAR (3D PANEL) ======== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(145deg, #fff3f8, #ffdce5);
        border-right: 3px solid #ffc4d1;
        box-shadow: 4px 0 25px rgba(255, 182, 193, 0.45);
        border-radius: 0 25px 25px 0;
    }

    /* ======== BUTTON ======== */
    .stButton>button {
        background: linear-gradient(145deg, #ff9ebd, #ff7aa8);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 10px 25px;
        font-weight: 600;
        box-shadow: 0 6px 12px rgba(255, 105, 180, 0.4);
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        background: linear-gradient(145deg, #ff7aa8, #ff4d91);
        box-shadow: 0 8px 20px rgba(255, 105, 180, 0.5);
    }

    /* ======== FILE UPLOADER (NEUMORPHIC BOX) ======== */
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #fff0f5, #ffd6e7);
        border-radius: 25px;
        border: 2px dashed #ff9bb8;
        padding: 25px;
        box-shadow: 6px 6px 15px rgba(255, 182, 193, 0.5),
                    -6px -6px 15px rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        transform: translateY(-2px);
        box-shadow: 8px 8px 20px rgba(255, 160, 176, 0.6),
                    -6px -6px 20px rgba(255, 255, 255, 0.9);
    }

    /* ======== ALERT CARD ======== */
    .stAlert {
        border-radius: 16px;
        background: linear-gradient(145deg, #fff0f6, #ffd7e2);
        box-shadow: 3px 3px 10px rgba(255, 182, 193, 0.4),
                    -3px -3px 10px rgba(255, 255, 255, 0.9);
    }

    /* ======== FOOTER ======== */
    footer {visibility: hidden;}
    .footer {
        text-align: center;
        padding: 15px;
        font-size: 14px;
        color: #8a0059;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# 🌸 LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Emmy Nora_Laporan 4.pt")  # model deteksi Spongebob vs Patrick
    classifier = tf.keras.models.load_model("model/Emmy Nora_Laporan2.h5")  # model klasifikasi Indoor vs Outdoor
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model kamu... tunggu sebentar ya 💕"):
    yolo_model, classifier = load_models()
st.success("✨ Model berhasil dimuat dengan sempurna! 🌸")

# ==========================
# 🌷 HEADER UTAMA
# ==========================
st.title("🌷 PinkVision: Cute Image & Object Detector 🌷")
st.markdown("""
Selamat datang di **PinkVision** 💖  
Aplikasi ini bisa melakukan dua hal utama:
- 🔍 **Deteksi objek (Spongebob vs Patrick)** menggunakan YOLO (.pt)  
- 🧠 **Klasifikasi gambar (Indoor vs Outdoor)** menggunakan model Keras (.h5)  
Unggah beberapa gambar sekaligus dengan **drag & drop** ya ✨
""")

# ==========================
# 🌸 SIDEBAR MENU
# ==========================
st.sidebar.header("🎀 Pilih Mode")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("💡 *Tips:* Gunakan gambar yang jelas biar hasil prediksi makin akurat 💕")

# ==========================
# 📸 UPLOAD GAMBAR
# ==========================
uploaded_files = st.file_uploader(
    "📸 Seret dan lepas (drag & drop) beberapa gambar di sini:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"🖼️ Total gambar diunggah: **{len(uploaded_files)} file**")

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=f"✨ {uploaded_file.name}", use_container_width=True)

        # ==========================
        # MODE DETEKSI OBJEK
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            with st.spinner(f"🔍 Mendeteksi objek pada {uploaded_file.name}..."):
                results = yolo_model.predict(img, conf=0.7, verbose=False)
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    st.image(results[0].plot(), caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                    st.success("✅ Objek terdeteksi dengan baik (Spongebob / Patrick)!")
                else:
                    st.warning("🚫 Tidak ada Spongebob atau Patrick yang terdeteksi.")
                    st.info("💡 Coba gambar lain yang mengandung karakter dari model.")

        # ==========================
        # MODE KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "Klasifikasi Gambar":
            with st.spinner(f"🧠 Mengklasifikasi {uploaded_file.name}..."):
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                labels = ["Indoor", "Outdoor"]
                predicted_label = labels[class_index]

                st.write(f"🎯 **Hasil Prediksi:** {predicted_label} ({confidence:.2f})")
                st.progress(float(confidence))

                if confidence > 0.85:
                    st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                elif confidence > 0.6:
                    st.warning("🌤️ Model agak ragu, tapi masih cukup yakin.")
                else:
                    st.error("😅 Model tidak yakin — mungkin ini bukan gambar indoor/outdoor.")
                    st.markdown("💡 **Saran:** Gunakan gambar ruangan atau lingkungan luar yang jelas 📷")

# ==========================
# 🌸 FOOTER
# ==========================
st.markdown("---")
st.markdown('<div class="footer">Made with 💕 by <b>Emmy Nora</b> 🌸</div>', unsafe_allow_html=True)

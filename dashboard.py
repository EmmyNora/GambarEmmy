import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="🌸 PinkVision: Smart & Cute AI 🌸",
    page_icon="💖",
    layout="wide"
)

# ==========================
# CUSTOM CSS (TEMA PINK)
# ==========================
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe4e9;
        color: #5c005c;
        font-family: "Poppins", sans-serif;
    }
    .stButton>button {
        background-color: #ff85a2;
        color: white;
        border-radius: 12px;
        padding: 8px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff4d79;
    }
    .stSidebar {
        background-color: #fff0f5;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Emmy Nora_Laporan 4.pt")  # model deteksi objek
    classifier = tf.keras.models.load_model("model/Emmy Nora_Laporan2.h5")  # model klasifikasi
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model kamu... tunggu sebentar ya 💕"):
    yolo_model, classifier = load_models()
st.success("✨ Model berhasil dimuat dengan sempurna! 🌸")

# ==========================
# HEADER
# ==========================
st.title("🌷 PinkVision: Cute Image & Object Detector 🌷")
st.markdown("""
Selamat datang di **PinkVision** 💖  
Aplikasi ini bisa melakukan:
- 🔍 Deteksi objek menggunakan **YOLO (.pt)**
- 🧠 Klasifikasi gambar menggunakan **Model Keras (.h5)**  
Unggah beberapa gambar sekaligus dengan **drag & drop** untuk hasil yang cepat dan lucu ✨
""")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("🎀 Pilih Mode")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("Cukup *drag & drop* gambar kamu ke kotak di bawah 💕")

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_files = st.file_uploader(
    "📸 Seret dan lepas (drag & drop) beberapa gambar di sini:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"🖼️ Total gambar diunggah: **{len(uploaded_files)} file**")

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"✨ {uploaded_file.name}", use_container_width=True)

        if menu == "Deteksi Objek (YOLO)":
            with st.spinner(f"🔍 Mendeteksi objek pada {uploaded_file.name}..."):
                results = yolo_model(img)
                result_img = results[0].plot()
                st.image(result_img, caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                st.success("✅ Deteksi selesai!")
                st.markdown("💡 **Tips:** Gunakan gambar dengan pencahayaan cukup agar hasil deteksi lebih akurat 🌞")

        elif menu == "Klasifikasi Gambar":
            with st.spinner(f"🧠 Mengklasifikasi {uploaded_file.name}..."):
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                labels = ["Indoor", "Outdoor"]  # sesuaikan dengan modelmu
                predicted_label = labels[class_index]

                st.write(f"🎯 **Hasil Prediksi:** {predicted_label}")
                st.progress(float(confidence))

                if confidence > 0.85:
                    st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                elif confidence > 0.6:
                    st.warning("🌤️ Model agak ragu, tapi masih cukup yakin.")
                else:
                    st.error("😅 Model kurang yakin. Coba gambar lain yang lebih jelas ya!")

                st.markdown("💡 **Saran:** Gunakan gambar fokus dan tidak blur agar hasil lebih akurat 📷")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<center>Made with 💕 by <b>Emmy Nora</b> 🌸</center>", unsafe_allow_html=True)

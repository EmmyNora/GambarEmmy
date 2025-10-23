import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# 🌸 KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="🌸 PinkVision: Smart & Cute AI 🌸",
    page_icon="💖",
    layout="wide"
)

# ==========================
# 🌷 CUSTOM CSS (TEMA PINK 3D + ANIMASI)
# ==========================
st.markdown("""
<style>
/* ======== BACKGROUND GRADIENT ======== */
.stApp {
    background: linear-gradient(to bottom, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
    overflow: hidden;
}

/* ======== SIDEBAR ======== */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe3eb, #ffc6d5, #ff9ec4);
    color: #4a0032;
    border-right: 3px solid #ff82a9;
    box-shadow: 4px 0 15px rgba(255, 100, 150, 0.3);
    padding-top: 1rem;
    border-radius: 0 20px 20px 0;
}

/* ======== TITLE ======== */
.main-title {
    text-align: center;
    font-size: 2.5rem;
    color: #b3005a;
    font-weight: 800;
    text-shadow: 2px 2px 6px #ffbad5;
    margin-top: 1.5rem;
}

/* ======== UPLOAD BOX ======== */
.upload-box {
    border: 3px dashed #ff8fab;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    background-color: rgba(255, 255, 255, 0.6);
    margin: 20px 0;
    box-shadow: 0 0 15px rgba(255, 150, 180, 0.3);
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
}

/* ======== FLOATY ANIMATION (CLOUD / FLOWER / JELLY) ======== */
@keyframes floaty {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-15px);}
    100% {transform: translateY(0px);}
}
.cloud, .flower, .jelly {
    position: absolute;
    opacity: 0.8;
    animation: floaty 6s ease-in-out infinite;
}
.cloud { width: 100px; top: 100px; left: 50%; transform: translateX(-50%); }
.flower { width: 80px; top: 60px; left: 25%; animation-delay: 1s; }
.jelly { width: 90px; bottom: 40px; right: 8%; animation-delay: 2s; }

/* ======== FOOTER ======== */
footer {visibility: hidden;}
.footer {
    text-align: center;
    color: #8a0059;
    padding: 15px;
    font-weight: 500;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# Animasi hiasan
st.markdown("""
<img src="https://i.ibb.co/5YxN9dP/cloud.png" class="cloud">
<img src="https://i.ibb.co/92CfksB/flower.png" class="flower">
<img src="https://i.ibb.co/JxkYk4Z/jelly.png" class="jelly">
""", unsafe_allow_html=True)

# ==========================
# 🌸 LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Emmy Nora_Laporan 4.pt")  # YOLO model
    classifier = tf.keras.models.load_model("model/Emmy Nora_Laporan2.h5")  # Keras model
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model... sabar ya 💕"):
    yolo_model, classifier = load_models()
st.success("✨ Model berhasil dimuat! 🌸")

# ==========================
# 🌷 HEADER
# ==========================
st.markdown('<div class="main-title">💗 PinkVision: Cute Image & Object Detector 💗</div>', unsafe_allow_html=True)
st.markdown("""
Selamat datang di *PinkVision* 💖  
Aplikasi ini bisa:
- 🔍 Deteksi objek **(Spongebob / Patrick)** menggunakan model YOLO (.pt)  
- 🧠 Klasifikasi gambar **(Indoor / Outdoor)** menggunakan model Keras (.h5)  
Unggah beberapa gambar dengan *drag & drop* ya ✨
""")

# ==========================
# 🌸 SIDEBAR
# ==========================
st.sidebar.markdown('<h3 style="color:#b3005a;">🎀 Pilih Mode</h3>', unsafe_allow_html=True)
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Gunakan gambar yang jelas agar hasil prediksi akurat 💕")

# ==========================
# 📸 UPLOAD GAMBAR
# ==========================
st.markdown('<div class="upload-box">📸 <b>Seret dan lepas</b> gambar kamu di sini 💕</div>', unsafe_allow_html=True)
uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"✨ {len(uploaded_files)} gambar berhasil diunggah!")
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        st.image(img, caption=file.name, use_container_width=True)

        # ========== YOLO DETEKSI ==========
        if mode == "Deteksi Objek (YOLO)":
            with st.spinner(f"🔍 Mendeteksi objek di {file.name}..."):
                results = yolo_model.predict(img, conf=0.7, verbose=False)
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    st.image(results[0].plot(), caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                    st.success("✅ Objek terdeteksi (Spongebob / Patrick)!")
                else:
                    st.warning("🚫 Tidak ada objek terdeteksi.")
                    st.info("💡 Coba gambar lain yang mengandung karakter dari model.")

        # ========== KLASIFIKASI GAMBAR ==========
        elif mode == "Klasifikasi Gambar":
            with st.spinner(f"🧠 Mengklasifikasi {file.name}..."):
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

                labels = ["Indoor", "Outdoor"]
                predicted_label = labels[class_index]

                st.write(f"🎯 *Hasil Prediksi:* {predicted_label} ({confidence:.2f})")
                st.progress(float(confidence))

                if confidence > 0.85:
                    st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                elif confidence > 0.6:
                    st.warning("🌤 Model agak ragu, tapi masih cukup yakin.")
                else:
                    st.error("😅 Model tidak yakin — mungkin ini bukan gambar indoor/outdoor.")
                    st.markdown("💡 *Saran:* Gunakan gambar ruangan atau luar ruangan yang jelas 📷")

# ==========================
# 🌸 FOOTER
# ==========================
st.markdown("---")
st.markdown('<div class="footer">Made with 💕 by <b>Emmy Nora</b> 🌸</div>', unsafe_allow_html=True)

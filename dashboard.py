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
    page_title="💗 PinkLens: Deteksi Objek & Klasifikasi Gambar 💗",
    page_icon="🌸",
    layout="wide"
)

# ==========================
# STYLE 
# ==========================
st.markdown("""
<style>
/* Background gradiasi lembut */
.stApp {
    background: linear-gradient(to bottom, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe3eb, #ffc6d5, #ffb3ca);
    color: #4a0032;
    border-right: 3px solid #ff82a9;
    box-shadow: 4px 0 15px rgba(255, 100, 150, 0.3);
    padding-top: 1rem;
}

/* Sidebar title */
.sidebar-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #b3005a;
    text-shadow: 1px 1px 3px #ffc0cb;
    margin-bottom: 1rem;
    text-align: center;
}

/* Deskripsi box di sidebar */
.desc-box {
    background-color: #ffe4ec;
    border: 2px solid #ff8ccf;
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 15px;
    color: #7b2c54;
    font-size: 14px;
    box-shadow: 0 3px 6px rgba(255, 182, 193, 0.4);
}
.desc-box b {
    color: #7b2c54;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 2.3rem;
    color: #b3005a;
    font-weight: 800;
    text-shadow: 2px 2px 6px #ffbad5;
    margin-top: 1rem;
}

/* Slogan */
.slogan {
    text-align: center;
    font-style: italic;
    color: #b3005a;
    font-size: 1.1rem;
    margin-bottom: 2.5rem; /* kasih jarak lebih besar */
}

/* Upload container */
.upload-container {
    border: 3px dashed #ff8fab;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    background-color: rgba(255, 255, 255, 0.5);
    margin: 0 auto;
    width: 80%;
    box-shadow: 0px 0px 15px rgba(255, 150, 180, 0.3);
}

/* Footer */
.footer {
    text-align: center;
    color: #b3005a;
    font-weight: 500;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Emmy Nora_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Emmy Nora_Laporan2.h5")
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model kamu... tunggu sebentar ya 💕"):
    yolo_model, classifier = load_models()

# ==========================
# SIDEBAR
# ==========================
st.sidebar.markdown('<div class="sidebar-title">🌸 Pilih Mode</div>', unsafe_allow_html=True)
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div class="desc-box">
        🔍 <b>Model YOLO (.pt)</b><br>
        mendeteksi karakter:<br>
        • 🟡 <b>Spongebob</b><br>
        • 💗 <b>Patrick</b>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div class="desc-box">
        🧠 <b>Model Keras (.h5)</b><br>
        mengklasifikasikan gambar:<br>
        • 🪴 <b>Indoor</b><br>
        • 🌤️ <b>Outdoor</b>
    </div>
    """, unsafe_allow_html=True)

# ==========================
# MAIN CONTENT
# ==========================
st.markdown('<div class="main-title">💗 PinkLens: Deteksi Objek & Klasifikasi Gambar 💗</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan">🌸 See Differently, See in Pink 🌸</div>', unsafe_allow_html=True)

# === Kotak upload ===
st.markdown("""
<div class="upload-container">
📸 <b>Seret dan lepas (drag & drop)</b> beberapa gambar kamu di sini 💕  
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Tombol prediksi
if uploaded_files:
    st.success(f"✨ {len(uploaded_files)} gambar berhasil diunggah!")
    if st.button("💖 Jalankan Prediksi / Klasifikasi 💖"):
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            st.image(img, caption=f"🖼️ {file.name}", use_container_width=True)

            # === MODE 1: DETEKSI OBJEK ===
            if mode == "Deteksi Objek (YOLO)":
                with st.spinner(f"🔍 Mendeteksi objek pada {file.name}..."):
                    results = yolo_model.predict(img, conf=0.6, verbose=False)
                    boxes = results[0].boxes

                    if boxes is not None and len(boxes) > 0:
                        result_img = results[0].plot()
                        st.image(result_img, caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                        st.success("✅ Objek berhasil terdeteksi!")
                    else:
                        st.warning("🚫 Tidak ada objek yang terdeteksi.")
                        st.info("💡 Coba gunakan gambar Spongebob atau Patrick untuk hasil terbaik.")

            # === MODE 2: KLASIFIKASI GAMBAR ===
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

                    if confidence >= 0.7:
                        st.write(f"🎯 *Hasil Prediksi:* **{predicted_label}** ({confidence:.2f})")
                        st.progress(float(confidence))
                        if confidence > 0.85:
                            st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                        elif confidence > 0.6:
                            st.warning("🌤 Model agak ragu, tapi masih cukup yakin.")
                    else:
                        st.error("😅 Model tidak yakin — mungkin ini bukan gambar indoor/outdoor.")
                        st.markdown("💡 *Saran:* Gunakan gambar yang lebih jelas 📷")

# ==========================
# FOOTER
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>Made 💕 by <b>Emmy Nora</b> 🌷</p>", unsafe_allow_html=True)

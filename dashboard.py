# ==========================
# 🌸 PINKLENS : DETEKSI OBJEK & KLASIFIKASI GAMBAR
# ==========================

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image


# ==========================
# 🌸 STYLE CSS
# ==========================
st.markdown("""
<style>
/* Background gradien pink 3D */
.stApp {
    background: linear-gradient(to bottom right, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
    color: #5a004e;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe6f2, #ffb6c1, #ff9ec4);
    color: #5a004e;
}
[data-testid="stSidebar"] * {
    color: #5a004e !important;
    font-family: 'Poppins', sans-serif;
}

/* Judul utama */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #ff3f8e;
    text-shadow: 2px 2px 10px rgba(255, 105, 180, 0.4);
    margin-top: 20px;
}

/* Slogan */
.slogan {
    text-align: center;
    font-size: 20px;
    color: #b3006b;
    font-style: italic;
    margin-bottom: 60px; /* jarak antara slogan dan upload box */
}

/* Kotak upload */
.upload-section {
    background: rgba(255, 255, 255, 0.6);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(255, 105, 180, 0.3);
}

/* Tombol */
button, .stButton>button {
    background-color: #ff7eb9;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 10px 24px;
    box-shadow: 0px 4px 10px rgba(255, 105, 180, 0.3);
    transition: 0.2s;
}
button:hover, .stButton>button:hover {
    background-color: #ff479c;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)


# ==========================
# 🌸 SIDEBAR
# ==========================
st.sidebar.title("🩷 Tentang PinkLens")
st.sidebar.markdown("""
**🔍 Model YOLO (.pt)**  
mendeteksi karakter:

- 🟡 **Spongebob**  
- 💗 **Patrick**

---

Pilih mode yang ingin dijalankan:
""")

mode = st.sidebar.radio("🎯 Pilih Mode", ["Deteksi Objek", "Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.markdown("✨ *See Differently, See in Pink* ✨")


# ==========================
# 🌸 HEADER UTAMA
# ==========================
st.markdown("<h1 class='title'>📷 PinkLens : Deteksi Objek & Klasifikasi Gambar</h1>", unsafe_allow_html=True)
st.markdown("<p class='slogan'>🌸 PinkLens: See Differently, See in Pink 🌸</p>", unsafe_allow_html=True)


# ==========================
# 📸 UPLOAD GAMBAR
# ==========================
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📸 Seret dan lepas (drag & drop) beberapa gambar di sini:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"🖼 Total gambar diunggah: *{len(uploaded_files)} file*")

    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=f"✨ {uploaded_file.name}", use_container_width=True)

    # ==========================
    # 🔮 TOMBOL PREDIKSI / DETEKSI
    # ==========================
    if st.button("🔍 Jalankan Prediksi"):
        st.write("🚀 Proses sedang berjalan...")

        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file).convert("RGB")

            if mode == "Deteksi Objek":
                # Contoh placeholder YOLO
                st.write(f"📦 Deteksi objek pada gambar **{uploaded_file.name}**:")
                st.success("✅ Ditemukan: Spongebob dan Patrick (contoh hasil)")
                # Model aslinya bisa ditambahkan begini:
                # model = YOLO("model_yolo.pt")
                # results = model(img)
                # st.image(results[0].plot(), caption="Hasil Deteksi", use_container_width=True)

            elif mode == "Klasifikasi Gambar":
                # Placeholder klasifikasi (contoh)
                st.write(f"🧠 Mengklasifikasikan gambar **{uploaded_file.name}**:")
                st.info("📊 Hasil: Gambar termasuk kategori 'Outdoor Scene' (contoh hasil)")
                # Contoh alur sebenarnya:
                # model = tf.keras.models.load_model("model_klasifikasi.h5")
                # img_resized = img.resize((224, 224))
                # x = image.img_to_array(img_resized)
                # x = np.expand_dims(x, axis=0)
                # x /= 255
                # preds = model.predict(x)
                # st.write("Hasil prediksi:", preds)

st.markdown("</div>", unsafe_allow_html=True)

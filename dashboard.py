import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

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
/* Background gradiasi */
.stApp {
    background: linear-gradient(to bottom, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe3eb, #ffc6d5, #ff9ec4);
    color: #4a0032;
    border-right: 3px solid #ff82a9;
    box-shadow: 4px 0 15px rgba(255, 100, 150, 0.3);
    padding-top: 1rem;
}

/* Title & slogan */
.main-title {
    text-align: center;
    font-size: 2.3rem;
    color: #b3005a;
    font-weight: 800;
    text-shadow: 2px 2px 6px #ffbad5;
    margin-top: 1rem;
}
.slogan {
    text-align: center;
    font-style: italic;
    color: #b3005a;
    font-size: 1.1rem;
    margin-bottom: 3rem;
}

/* Uploader box custom */
[data-testid="stFileUploader"] {
    border: 2px dashed #ff8fab;
    background-color: rgba(255, 240, 245, 0.7);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 0 15px rgba(255, 150, 180, 0.3);
    text-align: center;
    width: 80%;
    margin: 0 auto 2rem auto;
    transition: all 0.3s ease-in-out;
}
[data-testid="stFileUploader"]:hover {
    box-shadow: 0 0 25px rgba(255, 120, 160, 0.5);
    background-color: rgba(255, 245, 250, 0.9);
}

/* Text uploader */
[data-testid="stFileUploader"] section div {
    color: #b3005a !important;
    font-weight: 500 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #b3005a;
    font-weight: 500;
    margin-top: 4rem;
    padding-bottom: 1rem;
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
st.sidebar.title("🌸 Pilih Mode")
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# Deskripsi dinamis di sidebar
if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    ---
    ### 🎯 Tentang Mode Ini:
    Mode **Deteksi Objek (YOLO)** akan mencari dan mengenali objek yang ada di dalam gambar kamu 🧠✨  
    - Model: `YOLOv8`  
    - Gunakan gambar bertema **Spongebob vs Patrick** untuk hasil paling seru!  
    - Hasil: bounding box dan label objek yang terdeteksi 💕
    """)
else:
    st.sidebar.markdown("""
    ---
    ### 🧠 Tentang Mode Ini:
    Mode **Klasifikasi Gambar** digunakan untuk mengenali apakah gambar kamu bertema  
    **Indoor** atau **Outdoor** 🌇🌿  
    - Model: CNN berbasis `TensorFlow`  
    - Gunakan gambar ruangan atau pemandangan luar untuk uji coba!  
    - Hasil: label + tingkat keyakinan model 🎀
    """)

st.sidebar.markdown("""
---
💡 *Tips:*  
Kamu bisa upload lebih dari satu gambar sekaligus ya!  
Setelah itu klik tombol 💖 untuk memulai prediksi.
""")

# ==========================
# MAIN CONTENT
# ==========================
st.markdown('<div class="main-title">💗 PinkLens: Deteksi Objek & Klasifikasi Gambar 💗</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan">🌸 See Differently, See in Pink 🌸</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "📸 Seret dan lepas (drag & drop) beberapa gambar kamu di sini 💕",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Tombol prediksi
if uploaded_files:
    st.success(f"✨ {len(uploaded_files)} gambar berhasil diunggah!")
    if st.button("💖 Jalankan Prediksi / Klasifikasi 💖"):
        for file in uploaded_files:
            img = Image.open(file).convert("RGB")
            st.image(img, caption=f"🖼️ {file.name}", use_container_width=True)

            if mode == "Deteksi Objek (YOLO)":
                with st.spinner(f"🔍 Mendeteksi objek pada {file.name}..."):
                    results = yolo_model.predict(img, conf=0.6, verbose=False)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        st.image(results[0].plot(), caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                        st.success("✅ Objek berhasil terdeteksi!")
                    else:
                        st.warning("🚫 Tidak ada objek yang terdeteksi.")
                        st.info("💡 Coba gunakan gambar Spongebob atau Patrick untuk hasil terbaik.")
            else:
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

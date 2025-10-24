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
# STYLE GLOWING & FONT UNIK + SHADOW
# ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@700&family=Poppins:wght@400;600;800&display=swap');

.stApp {
    background: linear-gradient(to bottom right, #ffd6e0, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe6ee, #ffc6d5, #ff9ec4);
    border-right: 3px solid #ff82a9;
    box-shadow: 6px 0 25px rgba(255, 100, 150, 0.35);
    padding-top: 1rem;
}

/* Judul utama */
.main-title {
    text-align: center;
    font-size: 2.6rem;
    color: #b3005a;
    font-family: 'Comic Neue', cursive;
    text-shadow: 2px 2px 12px #ffbad5, 0 0 30px #ff8fab;
    margin-top: 1rem;
}

/* Slogan */
.slogan {
    text-align: center;
    font-style: italic;
    color: #b3005a;
    font-size: 1.2rem;
    margin-bottom: 3rem;
    text-shadow: 1px 1px 8px rgba(255,150,180,0.6);
}

/* Drag & Drop */
[data-testid="stFileUploader"] {
    border: 2px dashed #ff8fab;
    background-color: rgba(255, 240, 245, 0.8);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 0 25px rgba(255, 150, 180, 0.45);
    text-align: center;
    width: 80%;
    margin: 0 auto 2rem auto;
    transition: all 0.3s ease-in-out;
}
[data-testid="stFileUploader"]:hover {
    box-shadow: 0 0 40px rgba(255, 120, 160, 0.65);
    background-color: rgba(255, 250, 252, 0.95);
}

/* Tombol Jalankan */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #ff7eb9, #ff65a3, #ffb3c6);
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    border: none;
    border-radius: 20px;
    padding: 0.8rem 1.5rem;
    box-shadow: 0 0 25px rgba(255, 105, 180, 0.75), 0 0 50px rgba(255, 182, 193, 0.45);
    text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.8);
    transition: all 0.3s ease-in-out;
    font-family: 'Comic Neue', 'Poppins', cursive;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #ff99cc, #ff66b3, #ffb6c1);
    transform: scale(1.07);
    box-shadow: 0 0 45px rgba(255, 105, 180, 0.9), 0 0 70px rgba(255, 182, 193, 0.6);
}

/* Kartu hasil */
.result-card {
    background: rgba(255, 240, 245, 0.9);
    border-radius: 20px;
    padding: 1.3rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 25px rgba(255, 100, 150, 0.5), inset 0 0 15px rgba(255, 182, 193, 0.45);
    border: 1px solid #ff9ec4;
    transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
}
.result-card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 35px rgba(255, 130, 180, 0.6);
}

/* Footer */
.footer {
    text-align: center;
    color: #b3005a;
    font-weight: 600;
    font-family: 'Comic Neue', cursive;
    margin-top: 4rem;
    padding-bottom: 1rem;
    text-shadow: 1px 1px 8px rgba(255,150,180,0.6);
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

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div style='background-color:#ffe6ee; border-radius:15px; padding:15px; border:1px solid #ffb6c1; box-shadow:0 0 20px rgba(255,150,180,0.45); margin-top:1rem;'>
    <b>🔍 Model YOLO (.pt)</b><br>
    Mendeteksi karakter:<br>
    🧽 <b>Spongebob</b><br>
    ⭐ <b>Patrick</b>
    </div>
    """, unsafe_allow_html=True)
elif mode == "Klasifikasi Gambar":
    st.sidebar.markdown("""
    <div style='background-color:#ffe6ee; border-radius:15px; padding:15px; border:1px solid #ffb6c1; box-shadow:0 0 20px rgba(255,150,180,0.45); margin-top:1rem;'>
    <b>🧠 Model Klasifikasi</b><br>
    Mengenali jenis gambar:<br>
    🏠 <b>Indoor<br>
    🌳 <b>Outdoor</b>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("💡*Tips:* kamu bisa upload beberapa gambar sekaligus")

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

if uploaded_files:
    st.success(f"✨ {len(uploaded_files)} gambar berhasil diunggah!")

    if st.button(" Jalankan Prediksi / Klasifikasi "):
        # ==================== 1 GAMBAR ====================
        if len(uploaded_files) == 1:
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                st.image(img, caption=f"🖼️ {file.name}", use_container_width=True)

                if mode == "Deteksi Objek (YOLO)":
                    with st.spinner(f"🔍 Mendeteksi objek pada {file.name}..."):
                        results = yolo_model.predict(img, conf=0.6, verbose=False)
                        boxes = results[0].boxes

                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        if boxes is not None and len(boxes) > 0:
                            st.image(results[0].plot(), caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                            st.success("✅ Objek berhasil terdeteksi!")
                        else:
                            st.warning("🚫 Tidak ada objek yang terdeteksi.")
                            st.info("💡 Coba gunakan gambar Spongebob atau Patrick untuk hasil terbaik.")
                        st.markdown('</div>', unsafe_allow_html=True)

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

                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.write(f"🎯 *Hasil Prediksi:* **{predicted_label}** ({confidence:.2f})")
                        st.progress(float(confidence))
                        if confidence > 0.85:
                            st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                        elif confidence > 0.6:
                            st.warning("🌤 Model agak ragu, tapi masih cukup yakin.")
                        else:
                            st.error("😅 Model tidak yakin — mungkin ini bukan gambar indoor/outdoor.")
                            st.markdown("💡 *Saran:* Gunakan gambar yang lebih jelas 📷")
                        st.markdown('</div>', unsafe_allow_html=True)

        # ==================== BEBERAPA GAMBAR ====================
        else:
            cols = st.columns(2)
            for i, file in enumerate(uploaded_files):
                col = cols[i % 2]
                with col:
                    img = Image.open(file).convert("RGB")
                    st.image(img, caption=f"🖼️ {file.name}", use_container_width=True)

                    if mode == "Deteksi Objek (YOLO)":
                        with st.spinner(f"🔍 Mendeteksi objek pada {file.name}..."):
                            results = yolo_model.predict(img, conf=0.6, verbose=False)
                            boxes = results[0].boxes

                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            if boxes is not None and len(boxes) > 0:
                                st.image(results[0].plot(), caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                                st.success("✅ Objek berhasil terdeteksi!")
                            else:
                                st.warning("🚫 Tidak ada objek yang terdeteksi.")
                                st.info("💡 Coba gunakan gambar Spongebob atau Patrick untuk hasil terbaik.")
                            st.markdown('</div>', unsafe_allow_html=True)
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

                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            st.write(f"🎯 *Hasil Prediksi:* **{predicted_label}** ({confidence:.2f})")
                            st.progress(float(confidence))
                            if confidence > 0.85:
                                st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                            elif confidence > 0.6:
                                st.warning("🌤 Model agak ragu, tapi masih cukup yakin.")
                            else:
                                st.error("😅 Model tidak yakin — mungkin ini bukan gambar indoor/outdoor.")
                                st.markdown("💡 *Saran:* Gunakan gambar yang lebih jelas 📷")
                            st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# FOOTER
# ==========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='footer'>Made by <b>Emmy Nora</b> 🌷</p>", unsafe_allow_html=True)

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
.stApp {
    background: linear-gradient(to bottom, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe3eb, #ffc6d5, #ff9ec4);
    color: #4a0032;
    border-right: 3px solid #ff82a9;
    box-shadow: 4px 0 15px rgba(255, 100, 150, 0.3);
    padding-top: 1rem;
}
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
.result-card {
    background: rgba(255, 240, 245, 0.85);
    border-radius: 15px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 15px rgba(255, 100, 150, 0.3);
    border: 1px solid #ff9ec4;
}
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

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div style='background-color:#ffe6ee; border-radius:12px; padding:15px; border:1px solid #ffb6c1; margin-top:1rem;'>
    <b>🔍 Model YOLO (.pt)</b><br>
    Mendeteksi karakter:<br>
    • 🟡 <b>Spongebob</b><br>
    • 💗 <b>Patrick</b>
    </div>
    """, unsafe_allow_html=True)
elif mode == "Klasifikasi Gambar":
    st.sidebar.markdown("""
    <div style='background-color:#ffe6ee; border-radius:12px; padding:15px; border:1px solid #ffb6c1; margin-top:1rem;'>
    <b>🧠 Model Klasifikasi (.h5)</b><br>
    Mengenali jenis gambar:<br>
    • 🏠 <b>Indoor</b> — di dalam ruangan<br>
    • 🌳 <b>Outdoor</b> — di luar ruangan
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("💡 *Tips:* kamu bisa upload **beberapa gambar sekaligus** untuk deteksi & klasifikasi seru 💕")

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

    if st.button("💖 Jalankan Prediksi / Klasifikasi 💖"):
        # Jika hanya 1 gambar, tampilkan fullscreen
        if len(uploaded_files) == 1:
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                st.image(img, caption=f"🖼️ {file.name}", use_container_width=True)

                if mode == "Deteksi Objek (YOLO)":
                    with st.spinner(f"🔍 Mendeteksi objek pada {file.name}..."):
                        results = yolo_model.predict(img, conf=0.6, verbose=False)
                        boxes = results[0].boxes

                        with st.container():
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            if boxes is not None and len(boxes) > 0:
                                # tampilkan hasil dengan ukuran penuh
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

                        with st.container():
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

        # Jika lebih dari 1 gambar, tampilkan 2 kolom per baris
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
st.markdown("<p class='footer'>Made 💕 by <b>Emmy Nora</b> 🌷</p>", unsafe_allow_html=True)

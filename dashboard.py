import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import plotly.express as px

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="💖 PinkVision: Smart & Cute AI 💖",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
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
    classifier = tf.keras.models.load_model("model/Emmy Nora_Laporan2.h5")  # model klasifikasi utama
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model kamu... tunggu bentar ya 💕"):
    yolo_model, classifier = load_models()
st.success("Model berhasil dimuat! 🌸")

# ==========================
# HEADER
# ==========================
st.title("🌷 PinkVision: Cute Image & Object Detector 🌷")
st.markdown(
    "Selamat datang di **PinkVision**! 💖<br>"
    "Aplikasi ini bisa melakukan *deteksi objek (YOLO)*, *klasifikasi gambar*, "
    "*perbandingan dua model*, dan menampilkan *grafik akurasi* dengan gaya imut tapi cerdas 🧠✨",
    unsafe_allow_html=True
)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("🎀 Pilihan Mode")
menu = st.sidebar.selectbox(
    "Pilih Mode:",
    [
        "Deteksi Objek (YOLO)",
        "Klasifikasi Gambar",
        "Perbandingan Dua Model",
        "Grafik Akurasi Model"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar di bawah, lalu klik tombol *Mulai Prediksi!* 🌸")

# ==========================
# MODE: DETEKSI & KLASIFIKASI
# ==========================
if menu in ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]:
    uploaded_files = st.file_uploader(
        "📸 Unggah satu atau beberapa gambar kamu di sini:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption=f"✨ Gambar: {uploaded_file.name}", use_container_width=True)

            if st.button(f"🌷 Mulai Prediksi: {uploaded_file.name}"):
                with st.spinner("🪄 Sedang memproses..."):
                    time.sleep(1.2)

                    # ==========================
                    # MODE DETEKSI OBJEK
                    # ==========================
                    if menu == "Deteksi Objek (YOLO)":
                        results = yolo_model(img)
                        result_img = results[0].plot()
                        st.image(result_img, caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                        st.success("✨ Deteksi selesai dengan sukses! ✨")
                        st.markdown("💡 **Saran:** Jika hasil kurang akurat, coba gambar dengan pencahayaan lebih terang 🌞")

                    # ==========================
                    # MODE KLASIFIKASI
                    # ==========================
                    elif menu == "Klasifikasi Gambar":
                        img_resized = img.resize((128, 128))
                        img_array = image.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0

                        prediction = classifier.predict(img_array)
                        class_index = np.argmax(prediction)
                        confidence = np.max(prediction)

                        st.write(f"🎯 **Hasil Prediksi:** {class_index}")
                        st.progress(float(confidence))

                        if confidence > 0.85:
                            st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                        elif confidence > 0.6:
                            st.warning("🌤️ Model agak ragu, tapi masih cukup yakin.")
                        else:
                            st.error("😅 Model kurang yakin. Coba gambar lain yang lebih jelas ya!")

                        st.markdown("💡 **Saran:** Gunakan gambar jelas, tidak blur, agar hasil klasifikasi lebih akurat 📷")

# ==========================
# MODE: PERBANDINGAN DUA MODEL
# ==========================
if menu == "Perbandingan Dua Model":
    st.subheader("⚖️ Perbandingan Dua Model Klasifikasi")

    uploaded_file = st.file_uploader("📷 Unggah gambar untuk dibandingkan:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang akan diuji", use_container_width=True)

        modelA = tf.keras.models.load_model("model/Emmy Nora_Laporan2.h5")
        modelB = tf.keras.models.load_model("model/Model_Lain.h5")  # tambahkan model kedua kamu

        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predA = modelA.predict(img_array)
        predB = modelB.predict(img_array)

        classA, confA = np.argmax(predA), np.max(predA)
        classB, confB = np.argmax(predB), np.max(predB)

        st.write(f"🎀 **Model A (Laporan2)**: Prediksi {classA}, Kepercayaan {confA:.2f}")
        st.write(f"💫 **Model B (Lainnya)**: Prediksi {classB}, Kepercayaan {confB:.2f}")

        fig = px.bar(
            x=["Model A", "Model B"],
            y=[confA, confB],
            color=["Model A", "Model B"],
            text=[f"{confA:.2f}", f"{confB:.2f}"],
            title="📊 Perbandingan Tingkat Kepercayaan Model"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================
# MODE: GRAFIK AKURASI
# ==========================
if menu == "Grafik Akurasi Model":
    st.subheader("📊 Grafik Akurasi Model")
    model_names = ["Model A", "Model B", "Model C"]
    accuracy = [0.91, 0.88, 0.93]  # contoh data akurasi

    fig, ax = plt.subplots()
    ax.bar(model_names, accuracy, color=["#ff85a2", "#ffa6c9", "#ffb6d9"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Akurasi")
    ax.set_title("💖 Perbandingan Akurasi Model Klasifikasi 💖")

    st.pyplot(fig)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<center>Made with 💕 by Emmy Nora 🌸</center>", unsafe_allow_html=True)

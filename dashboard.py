import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ============================#
# 🌸 PAGE CONFIGURATION
# ============================#
st.set_page_config(
    page_title="PinkVision 💖",
    page_icon="💗",
    layout="wide",
)

# ============================#
# 🌷 CUSTOM CSS (Pink Style)
# ============================#
st.markdown("""
<style>
/* Background Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #ffd1dc, #ffb6c1);
    color: #4a4a4a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffe4ec;
    border-right: 3px solid #ff9eb5;
}

[data-testid="stSidebar"] h2 {
    color: #ff4b8b;
}

/* Section Box */
.main-container {
    background-color: #fff0f5;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(255, 100, 150, 0.2);
    text-align: center;
}

/* Header */
.title-text {
    font-size: 36px;
    color: #ff4b8b;
    text-shadow: 1px 1px 4px rgba(255, 80, 120, 0.3);
    font-weight: 800;
}

.subtitle-text {
    font-size: 16px;
    color: #6d4c57;
}

/* Footer */
.footer {
    text-align: center;
    color: #9b6b77;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ============================#
# 🌼 SIDEBAR
# ============================#
st.sidebar.markdown("## 🌸 Pilih Mode")
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

# ============================#
# 💗 FUNGSI: DETEKSI YOLO
# ============================#
def yolo_detection(uploaded_image):
    model = YOLO("model_yolo.pt")  # Ganti dengan path model kamu
    results = model(uploaded_image)
    for result in results:
        img_with_boxes = result.plot()  # Gambar hasil deteksi
        st.image(img_with_boxes, caption="Hasil Deteksi 💕", use_container_width=True)

# ============================#
# 💗 FUNGSI: KLASIFIKASI TENSORFLOW
# ============================#
def classify_image(uploaded_image):
    model = tf.keras.models.load_model("model_klasifikasi.h5")  # Ganti path model kamu
    img = uploaded_image.resize((224, 224))  # Sesuaikan resolusi model
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    st.success(f"🌼 Kelas Prediksi: {class_idx} (Kepercayaan: {confidence*100:.2f}%)")

# ============================#
# 🌷 MAIN CONTENT
# ============================#
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='title-text'>💖 PinkVision: Cute Image & Object Detector 💖</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Seret dan lepas (drag & drop) gambar kamu di bawah ini 💕</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Unggah gambar kamu di sini:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar kamu 💞", use_container_width=True)

    if mode == "Deteksi Objek (YOLO)":
        st.subheader("🔍 Hasil Deteksi Objek")
        yolo_detection(img)
    else:
        st.subheader("🧠 Hasil Klasifikasi Gambar")
        classify_image(img)
else:
    st.info("✨ Belum ada gambar yang diunggah")

st.markdown("</div>", unsafe_allow_html=True)

# ============================#
# 🌸 FOOTER
# ============================#
st.markdown("""
<div class='footer'>
    Made with 💕 by <b>Emmy Nora 🌷</b>
</div>
""", unsafe_allow_html=True)

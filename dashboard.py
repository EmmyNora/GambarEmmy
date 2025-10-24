import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ======== STYLE ========
st.markdown("""
<style>
/* Background gradiasi pink 3D */
.stApp {
    background: linear-gradient(to bottom, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe4ec, #ffd1dc, #ffb6c1);
    color: #4a4a4a;
    font-size: 15px;
}

/* Judul utama */
.title {
    text-align: center;
    color: #e75480;
    font-weight: 800;
    font-size: 36px;
}

/* Slogan */
.slogan {
    text-align: center;
    color: #ff69b4;
    font-style: italic;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Tombol */
.stButton > button {
    background-color: #ff80ab;
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}
.stButton > button:hover {
    background-color: #ff4081;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ======== TITLE ========
st.markdown('<h1 class="title">PinkLens : Deteksi Objek & Klasifikasi Gambar</h1>', unsafe_allow_html=True)
st.markdown('<p class="slogan">🌸 See Differently, See in Pink 🌸</p>', unsafe_allow_html=True)

# ======== SIDEBAR ========
st.sidebar.title("🔍 Model YOLO (.pt)")
st.sidebar.write("mendeteksi karakter:")
st.sidebar.markdown("""
- 🟡 **Spongebob**  
- 💗 **Patrick**
""")

# ======== UPLOAD IMAGE ========
uploaded_file = st.file_uploader("Unggah gambar untuk dideteksi atau diklasifikasi:", type=["jpg", "jpeg", "png"])

# ======== MODEL SETUP ========
# Load YOLO model
yolo_model = YOLO("model_yolo.pt")  # ganti dengan path model YOLO kamu
# Load CNN model (misal untuk klasifikasi tambahan)
cnn_model = tf.keras.models.load_model("model_klasifikasi.h5")  # ganti dengan modelmu

# ======== DETEKSI DAN KLASIFIKASI ========
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Tombol Prediksi
    if st.button("🔮 Deteksi / Klasifikasi"):
        with st.spinner("Model sedang bekerja... 💗"):
            # Deteksi dengan YOLO
            results = yolo_model(img)
            yolo_labels = results[0].boxes.cls if hasattr(results[0], 'boxes') else []
            st.subheader("🎯 Hasil Deteksi YOLO:")
            if len(yolo_labels) == 0:
                st.write("Tidak ada objek terdeteksi.")
            else:
                st.write(results[0].names)

            # Klasifikasi dengan CNN (contoh sederhana)
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            prediction = cnn_model.predict(img_array)
            pred_label = np.argmax(prediction)
            st.subheader("🧠 Hasil Klasifikasi CNN:")
            st.write(f"Label yang diprediksi: **{pred_label}**")

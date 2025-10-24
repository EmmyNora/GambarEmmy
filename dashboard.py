import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ======== PAGE CONFIG ========
st.set_page_config(
    page_title="PinkLens",
    page_icon="💗",
    layout="wide"
)

# ======== STYLE ========
st.markdown("""
<style>
/* Background gradien pink 3D */
.stApp {
    background: linear-gradient(to bottom, #ffdce5, #ffb6c1, #ff9ec4);
    font-family: 'Poppins', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe4ec, #ffc1d6);
    color: #4b0f31;
}

/* Sidebar title */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #b30059;
    margin-bottom: 10px;
}

/* Deskripsi box */
.desc-box {
    background-color: rgba(255, 255, 255, 0.5);
    border-radius: 15px;
    padding: 12px 15px;
    margin-top: 10px;
    box-shadow: 0px 3px 6px rgba(255, 182, 193, 0.4);
}

/* Tips */
.tips {
    font-size: 13px;
    color: #4b0f31;
    margin-top: 30px;
    padding-left: 8px;
}
</style>
""", unsafe_allow_html=True)

# ======== SIDEBAR ========
st.sidebar.markdown('<p class="sidebar-title">🌸 Pilih Mode</p>', unsafe_allow_html=True)
mode = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

if mode == "Deteksi Objek (YOLO)":
    st.sidebar.markdown("""
    <div class="desc-box">
        <b>🔍 Model YOLO (.pt)</b><br>
        Mendeteksi karakter:<br>
        • 🍍 <b>Spongebob</b><br>
        • 💗 <b>Patrick</b>
    </div>
    """, unsafe_allow_html=True)

elif mode == "Klasifikasi Gambar":
    st.sidebar.markdown("""
    <div class="desc-box">
        <b>🧠 Model Klasifikasi (.h5)</b><br>
        Mengenali jenis gambar:<br>
        • 🏠 <b>Indoor</b> — di dalam ruangan<br>
        • 🌳 <b>Outdoor</b> — di luar ruangan
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<p class="tips">💡 Kamu bisa upload <b>beberapa gambar sekaligus</b> 
untuk deteksi & klasifikasi seru 💕</p>
""", unsafe_allow_html=True)

# ======== MAIN AREA ========
st.markdown("""
<h1 style='text-align:center; color:#b30059;'>
💗 PinkLens: Deteksi Objek & Klasifikasi Gambar 💗
</h1>
<p style='text-align:center; color:#8b004f; font-style:italic;'>
🌸 See Differently, See in Pink 🌸
</p>
""", unsafe_allow_html=True)

# ======== FILE UPLOAD ========
uploaded_files = st.file_uploader(
    "📸 Seret dan lepas (drag & drop) beberapa gambar kamu di sini 💕",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

# ======== PROSES DETEKSI / KLASIFIKASI ========
if uploaded_files:
    if mode == "Deteksi Objek (YOLO)":
        model = YOLO("best.pt")  # ganti dengan modelmu

        if len(uploaded_files) == 1:
            # Jika hanya 1 gambar → tampil besar
            file = uploaded_files[0]
            img = Image.open(file)
            st.image(img, caption=file.name, use_container_width=True)
            
            results = model.predict(img)
            for r in results:
                result_img = r.plot()  # hasil deteksi
                st.image(result_img, caption="💖 Hasil Deteksi Objek 💖", use_container_width=True)
            st.success("✅ Objek berhasil terdeteksi!")

        else:
            # Jika lebih dari 1 gambar → tampil grid 2 kolom
            cols = st.columns(2)
            for idx, file in enumerate(uploaded_files):
                img = Image.open(file)
                results = model.predict(img)
                for r in results:
                    result_img = r.plot()
                    with cols[idx % 2]:
                        st.image(result_img, caption=f"💖 {file.name}", use_container_width=True)

    elif mode == "Klasifikasi Gambar":
        model = tf.keras.models.load_model("model_klasifikasi.h5")  # ganti model kamu
        class_names = ["Indoor", "Outdoor"]

        if len(uploaded_files) == 1:
            file = uploaded_files[0]
            img = Image.open(file)
            st.image(img, caption=file.name, use_container_width=True)

            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)
            label = class_names[int(pred[0] > 0.5)]

            st.subheader(f"💗 Gambar ini terklasifikasi sebagai: **{label}** 💗")

        else:
            cols = st.columns(2)
            for idx, file in enumerate(uploaded_files):
                img = Image.open(file)
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                pred = model.predict(img_array)
                label = class_names[int(pred[0] > 0.5)]
                with cols[idx % 2]:
                    st.image(img, caption=f"💗 {file.name} → {label}", use_container_width=True)

st.markdown("""
<p style='text-align:center; color:#8b004f; margin-top:30px;'>
Made 💕 by <b>Emmy Nora 🌷</b>
</p>
""", unsafe_allow_html=True)

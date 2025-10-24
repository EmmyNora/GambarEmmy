import streamlit as st
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
    overflow: hidden;
}

/* Sidebar style */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #ffe4ec, #ffc1d6);
    color: #4b0f31;
    font-family: 'Poppins', sans-serif;
}

/* Sidebar header */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #b30059;
    margin-bottom: 10px;
}

/* Mode section title */
.mode-section {
    font-size: 15px;
    font-weight: 600;
    color: #4b0f31;
    margin-top: 20px;
}

/* Deskripsi card */
.desc-box {
    background-color: rgba(255, 255, 255, 0.5);
    border-radius: 15px;
    padding: 12px 15px;
    margin-top: 10px;
    box-shadow: 0px 3px 6px rgba(255, 182, 193, 0.4);
}

/* Tips text */
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

# Deskripsi tampil sesuai mode yang dipilih
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

# Tips tanpa kotak
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

# Upload file area
uploaded_files = st.file_uploader(
    "📸 Seret dan lepas (drag & drop) beberapa gambar kamu di sini 💕",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption=file.name, use_container_width=True)

st.markdown("""
<p style='text-align:center; color:#8b004f; margin-top:30px;'>
Made 💕 by <b>Emmy Nora 🌷</b>
</p>
""", unsafe_allow_html=True)

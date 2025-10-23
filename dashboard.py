# ==========================
# CUSTOM CSS (3D PINK THEME)
# ==========================
st.markdown("""
    <style>
    /* ======== PAGE ======== */
    .stApp {
        background: linear-gradient(180deg, #ffe6f2, #ffd6e7);
        color: #5a004d;
        font-family: "Poppins", sans-serif;
    }

    /* ======== SIDEBAR ======== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff0f7, #ffd6e7);
        border-right: 3px solid #ffb6c1;
        box-shadow: 2px 0 15px rgba(255, 182, 193, 0.6);
    }
    .css-1d391kg {
        background-color: transparent !important;
    }

    /* ======== BUTTON ======== */
    .stButton>button {
        background: linear-gradient(145deg, #ff9ec4, #ff6fa5);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 10px 22px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(255, 105, 180, 0.4);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 105, 180, 0.6);
        background: linear-gradient(145deg, #ff80b0, #ff4d8d);
    }

    /* ======== FILE UPLOADER ======== */
    [data-testid="stFileUploader"] {
        background: #fff0f5;
        border: 2px dashed #ff91ae;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(255, 182, 193, 0.5);
        transition: 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        box-shadow: 0 6px 20px rgba(255, 105, 180, 0.5);
        transform: translateY(-3px);
    }

    /* ======== TITLE ======== */
    h1 {
        text-align: center;
        color: #800040;
        text-shadow: 1px 1px 3px rgba(255, 182, 193, 0.8);
    }

    /* ======== MESSAGE BOX ======== */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(255, 182, 193, 0.4);
    }

    /* ======== FOOTER ======== */
    footer {
        visibility: hidden;
    }
    .footer {
        text-align: center;
        padding: 20px 0;
        font-size: 14px;
        color: #7a004c;
    }
    </style>
""", unsafe_allow_html=True)

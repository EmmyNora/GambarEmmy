import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

st.markdown("""
    <style>
    /* ======== BASE STYLE ======== */
    .stApp {
        background: linear-gradient(180deg, #ffe9f0, #ffd6e7);
        color: #6a004f;
        font-family: "Poppins", sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

    /* ======== HEADER ======== */
    h1 {
        text-align: center;
        color: #8c005c;
        font-weight: 800;
        font-size: 2.4em;
        letter-spacing: 1px;
        text-shadow: 2px 2px 5px rgba(255, 182, 193, 0.6);
        margin-top: 10px;
    }

    /* ======== SIDEBAR (3D PANEL) ======== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(145deg, #fff3f8, #ffdce5);
        border-right: 3px solid #ffc4d1;
        box-shadow: 4px 0 25px rgba(255, 182, 193, 0.45);
        border-radius: 0 25px 25px 0;
    }

    /* ======== BUTTON ======== */
    .stButton>button {
        background: linear-gradient(145deg, #ff9ebd, #ff7aa8);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 10px 25px;
        font-weight: 600;
        box-shadow: 0 6px 12px rgba(255, 105, 180, 0.4);
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        background: linear-gradient(145deg, #ff7aa8, #ff4d91);
        box-shadow: 0 8px 20px rgba(255, 105, 180, 0.5);
    }

    /* ======== FILE UPLOADER (NEUMORPHIC BOX) ======== */
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #fff0f5, #ffd6e7);
        border-radius: 25px;
        border: 2px dashed #ff9bb8;
        padding: 25px;
        box-shadow: 6px 6px 15px rgba(255, 182, 193, 0.5),
                    -6px -6px 15px rgba(255, 255, 255, 0.8);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        transform: translateY(-2px);
        box-shadow: 8px 8px 20px rgba(255, 160, 176, 0.6),
                    -6px -6px 20px rgba(255, 255, 255, 0.9);
    }

    /* ======== CARD STYLE ======== */
    .stAlert {
        border-radius: 16px;
        background: linear-gradient(145deg, #fff0f6, #ffd7e2);
        box-shadow: 3px 3px 10px rgba(255, 182, 193, 0.4),
                    -3px -3px 10px rgba(255, 255, 255, 0.9);
    }

    /* ======== FOOTER ======== */
    footer {visibility: hidden;}
    .footer {
        text-align: center;
        padding: 15px;
        font-size: 14px;
        color: #8a0059;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

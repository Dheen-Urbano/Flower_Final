import os
import urllib.request
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_URL = "https://your-direct-download-link/model.h5"  # <-- replace this with your direct link
MODEL_PATH = "model.h5"
CLASS_NAMES_PATH = "class_names.npy"

@st.cache_data(show_spinner=False)
def download_file(url, filename):
    if not os.path.exists(filename):
        st.write(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    return filename

@st.cache_resource(show_spinner=False)
def load_my_model():
    model = load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAMES_PATH)
    return model, class_names

# Download model and class names if not present
download_file(MODEL_URL, MODEL_PATH)
download_file("https://your-direct-download-link/class_names.npy", CLASS_NAMES_PATH)  # update this too

model, class_names = load_my_model()

st.title("Image Classifier (Deep Learning)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (64, 64))
    st.image(img, channels="BGR", caption="Uploaded Image")

    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    preds = model.predict(img_input)
    pred_class = class_names[np.argmax(preds)]

    st.subheader(f"Prediction: {pred_class}")

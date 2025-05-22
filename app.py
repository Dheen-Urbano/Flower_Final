import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.title("Image Classifier (Deep Learning)")
model = load_model("model.h5")
class_names = np.load("class_names.npy")

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

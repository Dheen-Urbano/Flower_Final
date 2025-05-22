import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load model and class names
model = tf.keras.models.load_model("flower_model.h5")
class_names = np.load("class_names.npy")

st.title("Flower Classifier")

uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read image as array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocess
    img_resized = cv2.resize(img, (64, 64))
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Flower Class: **{predicted_class}**")

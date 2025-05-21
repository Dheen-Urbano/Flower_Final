import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

st.title("Flower Classifier 🌸")

# Load model and class names
model = load_model("flower_model.h5")
class_names = np.load("class_names.npy")

# Upload image
uploaded_file = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_resized = cv2.resize(image, (64, 64))
    st.image(image, caption="Uploaded Image", channels="BGR")

    img_array = image_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    st.subheader(f"Prediction: {class_names[class_index]}")


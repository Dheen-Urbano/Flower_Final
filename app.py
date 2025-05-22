import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import urllib.request
import os

# Download model if it doesn't exist
MODEL_URL = "https://github.com/Dheen-Urbano/Flower_Final/blob/main/flower_model.h5" 
MODEL_PATH = "flower_model.h5"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = load_model(MODEL_PATH)
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  
    return model, class_names

model, class_names = load_my_model()

st.title("Eye Disease Classification")
st.write("Upload an eye image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Show original image
    st.image(image, caption='Uploaded Image', channels="BGR")

    # Resize to expected input size
    expected_size = model.input_shape[1:3]  # (height, width)
    resized_image = cv2.resize(image, expected_size)

    # Convert to RGB and normalize
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(rgb_image / 255.0, axis=0)  # Shape: (1, H, W, 3)

    # Predict
    prediction = model.predict(input_image)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Class: **{predicted_class}**")

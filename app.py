import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model and class names
@st.cache_resource
def load_my_model():
    model = load_model("flower_model.h5")
    class_names = np.load("class_names.npy", allow_pickle=True)
    return model, class_names

model, class_names = load_my_model()

# Streamlit UI
st.title("Flower Classification App")
st.write("Upload an image of an flower to classify .")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Resize image to match model input
    resized_image = cv2.resize(image, (224, 224))  # Change if your model uses another size
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    
    # Normalize and expand dimensions
    input_image = np.expand_dims(rgb_image / 255.0, axis=0)
    
    # Prediction
    prediction = model.predict(input_image)
    predicted_class = class_names[np.argmax(prediction)]

    # Display image and result
    st.image(rgb_image, caption='Uploaded Image', use_column_width=True)
    st.success(f"Predicted Disease: **{predicted_class}**")

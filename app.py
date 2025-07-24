# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Load the trained model
model = load_model("model/gesture_model.h5")

# Get class labels from the dataset folder
gesture_classes = sorted(os.listdir("gesture_dataset"))

# Set Streamlit app title
st.title("✋ Hand Gesture Recognition")
st.write("Upload a hand gesture image and I'll predict the gesture.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = gesture_classes[np.argmax(prediction)]

    st.success(f"Predicted Gesture: **{predicted_class}**")

# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Load the trained model
@st.cache_resource
def load_gesture_model():
    model_path = "model/gesture_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

model = load_gesture_model()

# Get gesture class names from dataset folder
gesture_classes = sorted(os.listdir("gesture_dataset"))

# Streamlit App UI
st.title("🤖 Hand Gesture Recognition App")
st.write("Upload an image of a hand gesture, and the model will predict the gesture class.")

uploaded_file = st.file_uploader("📤 Upload a gesture image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((64, 64))  # match training image size
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if model:
        prediction = model.predict(img_array)
        predicted_class = gesture_classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"✅ Predicted Gesture: **{predicted_class}** ({confidence:.2f}%)")
    else:
        st.error("❌ Model not found. Please make sure `gesture_model.h5` is in the `model/` folder.")

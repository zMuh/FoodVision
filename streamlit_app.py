from ultralytics import YOLO
import tempfile
import streamlit as st
from PIL import Image

model = YOLO("./models/best.pt")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file:
    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # Run YOLO prediction
    results = model.predict(source=temp_path)

    # Display the original image
    st.image(Image.open(temp_path), caption="Uploaded Image")

    # Display YOLO result
    st.image(results[0].plot(), caption="Prediction")

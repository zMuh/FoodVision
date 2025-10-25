import streamlit as st
from model.model import predict

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file:
    result = predict(uploaded_file)
    st.image(result)

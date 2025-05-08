
import streamlit as st
from app.inference import run_inference

st.title("📝 Handwriting OCR System")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    with open("temp.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image("temp.png", caption="Uploaded Handwritten Image", use_column_width=True)
    result = run_inference("saved_model.h5", "temp.png")
    st.write("Predicted Text:", result)

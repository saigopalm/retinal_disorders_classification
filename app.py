import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch

@st.cache_resource
def load_model():
    model = YOLO("weights/best.pt")  # adjust if needed
    return model

model = load_model()

st.title("YOLOv8 Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        st.write("Classifying...")

        results = model(image)

        probs = results[0].probs
        class_id = int(probs.top1)
        class_name = model.names[class_id]
        confidence = probs.top1conf.item()

        st.markdown(f"### Prediction: `{class_name}`")
        st.markdown(f"**Confidence:** `{confidence:.2%}`")

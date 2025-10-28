import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import gdown

st.set_page_config(page_title="Multi-Component Detection", page_icon="📦")
st.title("📦 Multi-Component Detection")
st.write("Upload an image to detect multiple electronic components in it.")

# -------------------------
# Download a sample YOLOv8 small model if not exists (optional)
MODEL_PATH = "best.pt"
DRIVE_FILE_ID = "1TXKfQWxH4Jl5sJm3yX2nBz7R9uG2i5YA"  # Replace with your Drive file ID

if not os.path.exists(MODEL_PATH):
    st.info("Downloading YOLOv8 model...")
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# Load YOLOv8 model
st.info("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
st.success("Model loaded successfully!")

# -------------------------
# File upload
uploaded_file = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image_np, caption="Original Image", use_column_width=True)

    st.info("Detecting components...")
    results = model.predict(image_np)

    # Annotated image with bounding boxes and labels
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Detected Components", use_column_width=True)

    # Display detected class labels and confidence
    st.subheader("Detected Components:")
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.write("No components detected.")
    else:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"• {label} — Confidence: {conf:.2f}")

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Multi-Component Detection", page_icon="📦")
st.title("📦 Multi-Component Detection")
st.write("Upload an image to detect multiple electronic components in it.")

# -------------------------
# Load YOLOv8 model
MODEL_PATH = "best.pt"  # Replace with your trained YOLOv8 model
st.info("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
st.success("Model loaded successfully!")

# -------------------------
# Upload image
uploaded_file = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image_np, caption="Original Image", use_column_width=True)

    st.info("Detecting components...")
    results = model.predict(image_np)

    # Annotated image with boxes and labels
    annotated_image = results[0].plot()  # returns np.array with drawn boxes

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

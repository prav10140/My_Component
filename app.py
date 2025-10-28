import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown

st.title("🔲 Multi-Component Detection")

# Load model
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

model = tf.keras.models.load_model(MODEL_PATH)

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# Upload single image
uploaded_file = st.file_uploader("Upload an image", type=['png','jpg','jpeg'])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    orig = image.copy()
    h, w, _ = image.shape

    # Sliding window parameters
    window_size = 128
    step_size = 64  # overlap

    boxes = []
    labels = []

    for y in range(0, h - window_size + 1, step_size):
        for x in range(0, w - window_size + 1, step_size):
            patch = image[y:y+window_size, x:x+window_size]

            # Preprocess patch
            if patch.shape[-1] == 4:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            norm = gray / 255.0
            if np.mean(norm) > 0.5:
                gray = 255 - gray
            gray_resized = cv2.resize(gray, (128, 128))
            img_rgb = cv2.merge([gray_resized]*3)
            img_input = np.expand_dims(img_rgb / 255.0, axis=0)

            # Predict
            pred = model.predict(img_input)
            pred_class = np.argmax(pred)
            conf = np.max(pred)

            # If confidence > threshold, mark as detected
            if conf > 0.7:
                boxes.append((x, y, x+window_size, y+window_size))
                labels.append(class_labels[pred_class])

    # Draw boxes on original image
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(orig, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    st.image(orig, caption="Detected Components", use_column_width=True)

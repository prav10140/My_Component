import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import matplotlib.pyplot as plt

st.set_page_config(page_title="CNN Batch Prediction", page_icon="📷")

st.title("📷 CNN Batch Prediction")
st.write("Upload up to 6 images to get predictions from the CNN model.")

# -------------------------
# Download model from Google Drive if not exists
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded successfully!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# -------------------------
# Upload multiple images
uploaded = st.file_uploader("Upload up to 6 images", type=['png','jpg','jpeg'], accept_multiple_files=True)

if uploaded:
    n_files = min(len(uploaded), 6)
    st.write(f"Processing {n_files} image(s)...")
    
    fig, axes = plt.subplots(1, n_files, figsize=(4*n_files, 4))
    if n_files == 1:
        axes = [axes]  # Ensure axes is iterable
    
    for i, uploaded_file in enumerate(uploaded[:6]):
        # Read image with all channels
        img = np.array(Image.open(uploaded_file))

        # ✅ Convert RGBA → RGB if needed
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # ✅ Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ✅ Normalize to [0,1] before checking brightness
        norm = gray / 255.0

        # ✅ Invert if background is white
        if np.mean(norm) > 0.5:
            gray = 255 - gray

        # ✅ Resize & stack 3 channels
        gray_resized = cv2.resize(gray, (128, 128))
        img_rgb = cv2.merge([gray_resized]*3)
        img_input = np.expand_dims(img_rgb / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_input)
        pred_class = np.argmax(prediction)
        label = class_labels[pred_class]

        # Display
        axes[i].imshow(gray_resized, cmap='gray')
        axes[i].set_title(label, fontsize=12, color='green')
        axes[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

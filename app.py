import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown

st.set_page_config(page_title="CNN Image Prediction", page_icon="📷")
st.title("📷 CNN Image Prediction")
st.write("Upload images or take a photo using your back camera for prediction. On supported devices, flashlight can be turned on manually or automatically.")

# -------------------------
# Download model from Google Drive if not exists
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# -------------------------
# Two tabs: Camera or Upload
tab1, tab2 = st.tabs(["📸 Camera", "📁 Upload Images"])

# -------------------------
# Tab 1: Torch-enabled Camera Input
with tab1:
    st.markdown("""
    <p style='color:orange'>⚠️ On mobile, you can turn on the flashlight manually via your camera UI. Automatic torch may work on some devices only.</p>
    """, unsafe_allow_html=True)

    captured_image = st.camera_input("Take a photo (back camera works on mobile)")

    if captured_image is not None:
        # Preprocess
        img = np.array(Image.open(captured_image))
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm = gray / 255.0
        if np.mean(norm) > 0.5:
            gray = 255 - gray
        gray_resized = cv2.resize(gray, (128, 128))
        img_rgb = cv2.merge([gray_resized]*3)
        img_input = np.expand_dims(img_rgb / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_input)
        pred_class = np.argmax(prediction)
        label = class_labels[pred_class]

        st.image(gray_resized, caption=f"Prediction: {label}", use_column_width=True)

# -------------------------
# Tab 2: File Uploader
with tab2:
    uploaded = st.file_uploader(
        "Upload up to 6 images",
        type=['png','jpg','jpeg'],
        accept_multiple_files=True
    )

    if uploaded:
        n_files = min(len(uploaded), 6)
        st.write(f"Processing {n_files} image(s)...")
        cols = st.columns(n_files)

        for i, uploaded_file in enumerate(uploaded[:6]):
            # Preprocess
            img = np.array(Image.open(uploaded_file))
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            norm = gray / 255.0
            if np.mean(norm) > 0.5:
                gray = 255 - gray
            gray_resized = cv2.resize(gray, (128, 128))
            img_rgb = cv2.merge([gray_resized]*3)
            img_input = np.expand_dims(img_rgb / 255.0, axis=0)

            # Predict
            prediction = model.predict(img_input)
            pred_class = np.argmax(prediction)
            label = class_labels[pred_class]

            # Display
            with cols[i]:
                st.image(gray_resized, caption=f"Prediction: {label}", use_column_width=True)

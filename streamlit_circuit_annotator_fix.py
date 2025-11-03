# streamlit_circuit_annotator_fix.py
import os
import io
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Circuit Component Annotator (FIX)", page_icon="🔌")
st.title("🔌 Circuit Component Annotator — FIX")
st.write("Recommended: install `streamlit-drawable-canvas-fix` and run this script.")

# Defensive check: ensure the fixed canvas package is importable
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st.error(
        "Required package `streamlit-drawable-canvas-fix` is not installed in this environment.\n\n"
        "Install it with:\n\n    pip install streamlit-drawable-canvas-fix\n\n"
        "Then restart the app."
    )
    st.stop()

# ---------- SETTINGS ----------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"
DISPLAY_MAX_WIDTH = 900
INPUT_SIZE = (128, 128)

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# ---------- MODEL DOWNLOAD & LOAD ----------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded.")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# ---------- UI ----------
uploaded_file = st.file_uploader("Upload a full circuit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded_file:
    st.info("Upload a circuit image to begin.")
    st.stop()

orig_img = Image.open(uploaded_file).convert("RGB")
orig_w, orig_h = orig_img.size
display_w = min(DISPLAY_MAX_WIDTH, orig_w)
scale = orig_w / display_w
display_h = int(orig_h / scale)
disp_img = orig_img.resize((display_w, display_h))

st.caption("Draw a rectangle around ONE component, click 'Crop & Predict', rotate/preview, then Accept to stamp.")

# Use background_image (PIL Image) — the fixed package supports this cleanly
canvas_result = st_canvas(
    background_image=disp_img,
    stroke_width=2,
    stroke_color="#FF0000",
    update_streamlit=True,
    height=display_h,
    width=display_w,
    drawing_mode="rect",
    key="component_canvas_fix",
)

# Buttons to manage annotations
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Crop & Predict"):
        if canvas_result.json_data and "objects" in canvas_result.json_data and len(canvas_result.json_data["objects"])>0:
            obj = canvas_result.json_data["objects"][-1]
            left = int(obj["left"] * scale)
            top = int(obj["top"] * scale)
            width = int(obj["width"] * scale)
            height = int(obj["height"] * scale)
            left = max(0, left); top = max(0, top)
            right = min(orig_w, left + width); bottom = min(orig_h, top + height)
            if right - left <= 0 or bottom - top <= 0:
                st.error("Invalid bounding box — try again.")
            else:
                crop = orig_img.crop((left, top, right, bottom))
                st.session_state["last_crop"] = (crop, (left, top, right, bottom))
        else:
            st.warning("Draw a rectangle first.")
with col2:
    if st.button("Clear last rectangle"):
        st.experimental_rerun()
with col3:
    if st.button("Clear all annotations"):
        st.session_state["annotations"] = []
        st.experimental_rerun()

# Rotation & predict controls
if "last_crop" in st.session_state:
    crop_img, bbox = st.session_state["last_crop"]
    st.subheader("Preview & Predict")
    colA, colB = st.columns([1,1])
    with colA:
        angle = st.slider("Rotate crop (degrees)", -180, 180, 0)
        rotated = crop_img.rotate(angle, expand=True)
        st.image(rotated.resize((256,256)), caption="Rotated crop preview", use_column_width=False)

        with st.spinner("Preparing & predicting..."):
            arr = np.array(rotated.convert('L'))
            norm = arr / 255.0
            if np.mean(norm) > 0.5:
                arr = 255 - arr
            resized = cv2.resize(arr, INPUT_SIZE)
            stacked = np.stack([resized]*3, axis=-1)
            inp = np.expand_dims(stacked/255.0, axis=0)
            preds = model.predict(inp)
            pred_idx = int(np.argmax(preds))
            label = class_labels[pred_idx]
            conf = float(np.max(preds))

        if st.button(f"Accept and stamp as: {label} ({conf:.2f})"):
            if "annotations" not in st.session_state:
                st.session_state["annotations"] = []
            st.session_state["annotations"].append({
                "bbox": bbox, "label": label, "confidence": conf
            })
            del st.session_state["last_crop"]
            st.success(f"Annotated: {label}")

    with colB:
        st.write("[Model prediction details shown above]")

# Build annotated image and show
base = orig_img.copy()
draw = ImageDraw.Draw(base)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

if "annotations" in st.session_state and st.session_state["annotations"]:
    for ann in st.session_state["annotations"]:
        l,t,r,b = ann["bbox"]
        draw.rectangle([l,t,r,b], outline=(255,0,0), width=3)
        text = f"{ann['label']} ({ann['confidence']:.2f})"
        text_w, text_h = draw.textsize(text, font=font)
        draw.rectangle([l, t-text_h-6, l+text_w+6, t], fill=(255,0,0))
        draw.text((l+3, t-text_h-4), text, fill=(255,255,255), font=font)

st.subheader("Annotated circuit")
st.image(base, use_column_width=True)

buf = io.BytesIO()
base.save(buf, format="PNG")
st.download_button("Download annotated image (PNG)", data=buf.getvalue(), file_name="annotated_circuit.png", mime="image/png")

import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ==========================================================
# 🔧 STREAMLIT DRAWABLE CANVAS STABILITY PATCH
# ==========================================================
import streamlit.elements.image as st_image

if not hasattr(st_image, "image_to_url"):
    try:
        from streamlit.runtime.media_file_storage import media_file_storage
        from streamlit.elements.image import _normalize_to_bytes

        def image_to_url(image, width, clamp):
            image_bytes, mimetype = _normalize_to_bytes(image, width, clamp)
            file_id = media_file_storage.add(image_bytes, mimetype)
            return media_file_storage.get_url(file_id)

        st_image.image_to_url = image_to_url
    except Exception:
        pass

from streamlit_drawable_canvas import st_canvas

# ----------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------
st.set_page_config(page_title="Circuit Sketcher", page_icon="✏️")
st.title("✏️ Circuit Sketch-to-Component")
st.write("Draw a circuit component (Resistor, Capacitor, etc.) and let the AI identify it.")

# ----------------------------------------------------------
# MODEL CONFIG
# ----------------------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"
LABELS = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
except Exception as e:
    st.error("❌ Model failed to load")
    st.stop()

# ----------------------------------------------------------
# WHITEBOARD CANVAS
# ----------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Whiteboard")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # No fill
        stroke_width=3,
        stroke_color="#000000",               # Black ink
        background_color="#FFFFFF",           # White background
        height=400,
        width=400,
        drawing_mode="freedraw",              # Changed from 'rect' to 'freedraw'
        key="whiteboard",
    )

with col2:
    st.subheader("Controls")
    if st.button("🗑️ Clear Canvas"):
        st.rerun()
    
    analyze_button = st.button("🔍 Recognize Sketch")

# ----------------------------------------------------------
# PROCESSING
# ----------------------------------------------------------
if canvas_result.image_data is not None and analyze_button:
    # Get the image from canvas (RGBA)
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    # Convert to RGB (removes alpha)
    img_white = Image.new("RGB", img.size, (255, 255, 255))
    img_white.paste(img, mask=img.split()[3]) 
    
    # Find bounding box of the sketch to "auto-crop"
    # This removes empty white space around your drawing
    gray = img_white.convert("L")
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()

    if bbox:
        crop = img_white.crop(bbox)
        # Add small padding
        crop = ImageOps.expand(crop, border=20, fill="white")
        
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.image(crop, caption="Processed Sketch", width=200)
        
        with c2:
            # Prepare for AI
            resized = crop.resize((128, 128)).convert("RGB")
            arr = np.array(resized) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            preds = model.predict(arr)
            idx = np.argmax(preds)
            
            st.success(f"### Result: {LABELS[idx]}")
            st.progress(float(np.max(preds)))
            st.write(f"Confidence: {np.max(preds):.2%}")
    else:
        st.warning("Canvas is empty! Draw something first.")

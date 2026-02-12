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
st.title("✏️ Circuit Sketch Recognizer")
st.write("Draw a component (e.g., a resistor, capacitor, or diode) on the board and click 'Recognize'.")

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
except Exception:
    st.error("❌ Model failed to load. Check your internet connection or File ID.")
    st.stop()

# ----------------------------------------------------------
# WHITEBOARD INTERFACE
# ----------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Drawing Board")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=4,                # Thicker lines are better for AI
        stroke_color="#000000",       # Black ink
        background_color="#FFFFFF",   # White board
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="whiteboard",
    )

with col2:
    st.subheader("Actions")
    if st.button("🗑️ Clear Board"):
        st.rerun()
    
    analyze_button = st.button("🔍 Recognize Sketch")

# ----------------------------------------------------------
# AUTO-CROP & PREDICTION LOGIC
# ----------------------------------------------------------
if canvas_result.image_data is not None and analyze_button:
    # 1. Convert Canvas array to PIL Image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    # 2. Paste onto a solid white background (removes transparency)
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3]) 
    
    # 3. AUTO-CROP: Detect the drawing and trim edges
    gray = bg.convert("L")
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()

    if bbox:
        # Crop to the drawing and add 25px white padding
        final_crop = bg.crop(bbox)
        final_crop = ImageOps.expand(final_crop, border=25, fill="white")
        
        st.divider()
        c1, c2 = st.columns(2)
        
        with c1:
            st.image(final_crop, caption="What the AI sees", width=200)
        
        with c2:
            # Prepare for AI processing
            input_img = final_crop.resize((128, 128)).convert("RGB")
            arr = np.array(input_img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            # Predict
            preds = model.predict(arr)
            idx = np.argmax(preds)
            confidence = np.max(preds)
            
            st.success(f"### Result: {LABELS[idx]}")
            st.write(f"**Confidence:** {confidence:.2%}")
            st.progress(float(confidence))
    else:
        st.warning("The board is empty. Please draw a component first!")

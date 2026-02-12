import os
import io
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ==========================================
# 🔧 CRITICAL COMPATIBILITY PATCH
# ==========================================
import streamlit.elements.image as st_image
try:
    if not hasattr(st_image, 'image_to_url'):
        from streamlit.elements.utils import image_to_url
        st_image.image_to_url = image_to_url
except Exception:
    pass 

from streamlit_drawable_canvas import st_canvas
# ==========================================

st.set_page_config(page_title="Circuit Sketcher", page_icon="✏️")
st.title("✏️ Circuit Sketch Recognizer")
st.write("Draw a component (e.g., resistor, capacitor) and click **Recognize**.")

# ------------------------------------------
# MODEL LOADING
# ------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_" 

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model error: {e}")

# ------------------------------------------
# WHITEBOARD UI
# ------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=4,
        stroke_color="#000000",        # Black ink
        background_color="#FFFFFF",    # White board
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="whiteboard",
    )

with col2:
    st.write("### Actions")
    if st.button("🗑️ Clear"):
        st.rerun()
    
    analyze_btn = st.button("🔍 Recognize")

# ------------------------------------------
# CROP & PREDICT LOGIC
# ------------------------------------------
if canvas_result.image_data is not None and analyze_btn:
    # 1. Convert Canvas to PIL Image
    raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
    
    # 2. Paste on white background
    white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
    white_bg.paste(raw_img, mask=raw_img.split()[3]) 
    
    # 3. AUTO-CROP: Detect drawing edges
    gray = white_bg.convert("L")
    inverted = ImageOps.invert(gray)
    bbox = inverted.getbbox()

    if bbox:
        final_input = white_bg.crop(bbox)
        final_input = ImageOps.expand(final_input, border=30, fill="white")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.image(final_input, caption="Processed Sketch", width=200)
        with c2:
            prep = final_input.resize((128, 128)).convert("RGB")
            arr = np.array(prep) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            preds = model.predict(arr)
            labels = ['Ammeter', 'ac_src', 'battery', 'cap', 'curr_src', 'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src', 'dep_volt', 'diode', 'gnd_1', 'gnd_2', 'inductor', 'resistor', 'voltmeter']
            st.success(f"### Result: {labels[np.argmax(preds)]}")
            st.progress(float(np.max(preds)))
    else:
        st.warning("Draw something first!")

import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ==========================================
# 🛡️ THE ULTIMATE STABILITY PATCH
# ==========================================
import streamlit.elements.image as st_image
if not hasattr(st_image, 'image_to_url'):
    try:
        from streamlit.elements.utils import image_to_url
        st_image.image_to_url = image_to_url
    except Exception:
        def dummy(data, width, height, clamp, channels, output_format, image_id): return ""
        st_image.image_to_url = dummy

from streamlit_drawable_canvas import st_canvas
# ==========================================

st.set_page_config(page_title="Circuit Sketcher", page_icon="✏️")
st.title("✏️ Simple Whiteboard Detection")
st.write("Draw a component in the box and click **Analyze Sketch**.")

# --- MODEL LOADING ---
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_" 

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"AI Loading Error: {e}")
    st.stop()

LABELS = ['Ammeter', 'ac_src', 'battery', 'cap', 'curr_src', 'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src', 'dep_volt', 'diode', 'gnd_1', 'gnd_2', 'inductor', 'resistor', 'voltmeter']

# --- WHITEBOARD ---
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)", 
    stroke_width=4,
    stroke_color="#000000", # User draws in BLACK
    background_color="#FFFFFF", # Background is WHITE
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="board",
)

if st.button("🔍 Analyze Sketch"):
    if canvas_result.image_data is not None:
        # 1. Convert drawing to PIL RGB
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3]) 
        
        # 2. Convert to Grayscale and INVERT (This creates white lines on black)
        # Your model expects the format seen in your second image.
        gray_img = bg.convert("L")
        inverted_img = ImageOps.invert(gray_img) # Lines become WHITE, background becomes BLACK
        
        # 3. Auto-Crop drawing based on the inverted image
        bbox = inverted_img.getbbox()
        
        if bbox:
            crop = inverted_img.crop(bbox)
            crop = ImageOps.expand(crop, border=25, fill=0) # Padding with BLACK
            
            st.image(crop, caption="Processed Image for Model", width=150)
            
            # 4. Final Prediction Processing
            prep = crop.resize((128, 128))
            arr = np.array(prep) / 255.0
            arr = np.expand_dims(arr, axis=-1) # Add channel for Grayscale
            arr = np.stack([arr.squeeze()]*3, axis=-1) # Match 3-channel RGB expectation
            arr = np.expand_dims(arr, axis=0)
            
            res = model.predict(arr)
            idx = np.argmax(res)
            st.success(f"### Result: {LABELS[idx]}")
            st.info(f"Confidence: {np.max(res):.2%}")
        else:
            st.warning("Please draw a component on the whiteboard first!")

import os
import io
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps, ImageDraw

# ==========================================
# 🛡️ STABILITY PATCH
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

st.set_page_config(page_title="Sharp Circuit AI", page_icon="🔌")
st.title("🔌 Sharp Multi-Component Sketcher")
st.write("Draw components with **freedraw**, then use **rect** to box them for identification.")

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

# --- UI ---
mode = st.radio("Tool", ("freedraw", "rect"), horizontal=True)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.2)", 
    stroke_width=4, # Thicker strokes = Better sharpening
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=500,
    width=800,
    drawing_mode=mode,
    key="pro_board",
)

if canvas_result.json_data:
    objects = canvas_result.json_data.get("objects", [])
    rects = [obj for obj in objects if obj['type'] == 'rect']
    
    if rects and st.button("🔍 Analyze Selected Boxes"):
        # 1. Capture drawing
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg.paste(raw_img, mask=raw_img.split()[3]) 
        
        # 2. SHARPENING: Convert to 100% Black and White
        full_gray = np.array(white_bg.convert("L"))
        # This makes lines bright white and background deep black
        sharp_bw = cv2.adaptiveThreshold(full_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        sharp_pil = Image.fromarray(sharp_bw)

        labeled_img = white_bg.copy()
        draw = ImageDraw.Draw(labeled_img)
        
        st.divider()
        st.subheader("Results")
        cols = st.columns(min(len(rects), 4))

        for i, rect in enumerate(rects):
            l, t, w, h = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
            r, b = l + w, t + h
            
            # 3. Predict from the SHARPENED version
            crop = sharp_pil.crop((l, t, r, b))
            input_arr = np.array(crop.resize((128, 128)).convert("RGB")) / 255.0
            preds = model.predict(np.expand_dims(input_arr, axis=0))
            
            label = LABELS[np.argmax(preds)]
            conf = np.max(preds)
            
            # 4. Stamp and Display
            draw.rectangle([l, t, r, b], outline="red", width=5)
            draw.text((l + 10, t + 10), label, fill="red")
            
            with cols[i % 4]:
                st.image(crop, caption=f"Sharp Input {i+1}", use_column_width=True)
                st.write(f"**{label}**")

        st.image(labeled_img, caption="Final Labeled Board", use_column_width=True)

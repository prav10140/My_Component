import os
import io
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont

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

st.set_page_config(page_title="Sharp AI Sketcher", page_icon="🔌")
st.title("🔌 Professional Circuit Sketcher")
st.write("Draw components with **freedraw**, then use **rect** to box and identify them.")

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
    st.error(f"AI Error: {e}")
    st.stop()

LABELS = ['Ammeter', 'ac_src', 'battery', 'cap', 'curr_src', 'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src', 'dep_volt', 'diode', 'gnd_1', 'gnd_2', 'inductor', 'resistor', 'voltmeter']

# --- UI CONTROLS ---
col1, col2 = st.columns([1, 1])
with col1:
    mode = st.radio("Drawing Tool", ("freedraw", "rect"), horizontal=True)
with col2:
    if st.button("🗑️ Clear Whiteboard"):
        st.rerun()

# --- THE WHITEBOARD ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.2)", 
    stroke_width=4, # Thicker strokes for sharper lines
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=500,
    width=800,
    drawing_mode=mode,
    key="pro_multi_board",
)

# --- PROCESSING ---
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    rect_boxes = [obj for obj in objects if obj['type'] == 'rect']
    
    if rect_boxes and st.button("🔍 Analyze Selection Boxes"):
        # 1. Capture the sketch from canvas
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg.paste(raw_img, mask=raw_img.split()[3]) 
        
        # Pre-convert canvas to sharp black-and-white
        full_gray = np.array(white_bg.convert("L"))
        # Adaptive threshold makes the lines very sharp and bright
        thresh = cv2.adaptiveThreshold(full_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        sharp_base = Image.fromarray(thresh)

        # Output Setup
        final_labeled = white_bg.copy()
        draw = ImageDraw.Draw(final_labeled)
        
        st.divider()
        st.subheader("Sharp Component Previews")
        cols = st.columns(min(len(rect_boxes), 4))

        for i, rect in enumerate(rect_boxes):
            l, t, w, h = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
            r, b = l + w, t + h
            
            # 2. Extract the sharp crop for the model
            crop = sharp_base.crop((l, t, r, b))
            
            # 3. Model Preparation
            model_ready = crop.resize((128, 128)).convert("RGB")
            input_arr = np.array(model_ready) / 255.0
            input_arr = np.expand_dims(input_arr, axis=0)
            
            # 4. Predict
            preds = model.predict(input_arr)
            idx = np.argmax(preds)
            label = LABELS[idx]
            conf = np.max(preds)
            
            # 5. Visual Stamp
            draw.rectangle([l, t, r, b], outline="red", width=5)
            draw.text((l + 10, t + 10), f"{label}", fill="red")
            
            with cols[i % 4]:
                st.image(crop, caption=f"Sharp AI Crop {i+1}", use_column_width=True)
                st.write(f"**{label}**")

        st.divider()
        st.subheader("Full Labeled Circuit")
        st.image(final_labeled, use_column_width=True)

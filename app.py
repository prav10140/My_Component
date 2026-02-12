import os
import io
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
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

st.set_page_config(page_title="Multi-Component Sketcher", page_icon="🔌")
st.title("🔌 Multi-Component AI Sketcher")
st.write("1. **Freedraw**: Sketch your components. 2. **Rect**: Draw boxes around them to identify.")

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

# --- UI CONTROLS ---
col1, col2 = st.columns([1, 1])
with col1:
    mode = st.radio("Drawing Mode", ("freedraw", "rect"), help="Sketch first, then box for analysis.")
with col2:
    if st.button("🗑️ Reset Board"):
        st.rerun()

# --- THE WHITEBOARD ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.2)", 
    stroke_width=3,
    stroke_color="#000000", # User sketches in Black
    background_color="#FFFFFF", # Canvas is White
    height=500,
    width=700,
    drawing_mode=mode,
    key="multi_board_final",
)

# --- MULTI-CROP & LABELING LOGIC ---
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    rect_boxes = [obj for obj in objects if obj['type'] == 'rect']
    
    if rect_boxes and st.button("🔍 Analyze & Label All"):
        # Create base image from the canvas sketches
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        white_bg_canvas = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg_canvas.paste(raw_img, mask=raw_img.split()[3]) 
        
        # Output setup
        labeled_output = white_bg_canvas.copy()
        draw = ImageDraw.Draw(labeled_output)
        
        st.divider()
        st.subheader("Component Breakdowns")
        cols = st.columns(min(len(rect_boxes), 4))

        for i, rect in enumerate(rect_boxes):
            l, t, w, h = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
            r, b = l + w, t + h
            
            # --- CRITICAL PREPROCESSING FOR ACCURACY ---
            # 1. Crop the sketch
            crop = white_bg_canvas.crop((l, t, r, b))
            
            # 2. Convert to Grayscale and INVERT
            # This turns Black lines on White -> White lines on Black (Matching your model)
            processed_crop = ImageOps.invert(crop.convert("L"))
            
            # 3. Resize and Normalization
            model_input = processed_crop.resize((128, 128)).convert("RGB")
            input_arr = np.array(model_input) / 255.0
            input_arr = np.expand_dims(input_arr, axis=0)
            
            # 4. Predict
            predictions = model.predict(input_arr)
            idx = np.argmax(predictions)
            label = LABELS[idx]
            confidence = np.max(predictions)
            
            # 5. Stamp Labels on the Output
            draw.rectangle([l, t, r, b], outline="red", width=4)
            draw.text((l + 5, t + 5), f"{label}", fill="red")
            
            # Display result in grid
            with cols[i % 4]:
                st.image(processed_crop, caption=f"Box {i+1}: {label}", width=120)

        st.divider()
        st.subheader("Final Labeled Result")
        st.image(labeled_output, use_column_width=True)

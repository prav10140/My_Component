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
st.title("🔌 Multi-Component Sketcher & Labeler")
st.write("1. Draw your components. 2. Switch mode to 'Rect' to box them for analysis.")

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
    mode = st.radio("Drawing Mode", ("freedraw", "rect"), help="Use freedraw to sketch, then rect to select for AI.")
with col2:
    if st.button("🗑️ Reset All"):
        st.rerun()

# --- THE WHITEBOARD ---
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.2)", 
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=500,
    width=700,
    drawing_mode=mode,
    key="multi_board",
)

# --- PROCESSING MULTIPLE CROPS ---
if canvas_result.json_data is not None:
    objects = canvas_result.json_data.get("objects", [])
    rects = [obj for obj in objects if obj['type'] == 'rect']
    
    if rects and st.button("🔍 Analyze All Boxes"):
        # Create the base image from the canvas (the sketches)
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        white_bg = Image.new("RGB", raw_img.size, (255, 255, 255))
        white_bg.paste(raw_img, mask=raw_img.split()[3]) 
        
        # Prepare to draw labels on the final result
        final_draw_img = white_bg.copy()
        draw = ImageDraw.Draw(final_draw_img)
        try:
            font = ImageFont.load_default()
        except:
            font = None

        st.divider()
        st.subheader("Analysis Results")
        
        for i, rect in enumerate(rects):
            # 1. Get Crop Coordinates
            l, t, w, h = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
            r, b = l + w, t + h
            
            # 2. Extract and Invert (White lines on Black for Model)
            crop = white_bg.crop((l, t, r, b))
            gray_crop = crop.convert("L")
            inverted_crop = ImageOps.invert(gray_crop)
            
            # 3. Predict
            prep = inverted_crop.resize((128, 128)).convert("RGB")
            arr = np.array(prep) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            res = model.predict(arr)
            idx = np.argmax(res)
            label = LABELS[idx]
            conf = np.max(res)
            
            # 4. Draw Label on Final Image
            draw.rectangle([l, t, r, b], outline="red", width=3)
            draw.text((l, t - 15), f"{label} ({conf:.1%})", fill="red", font=font)
            
            # Show individual crops in a grid
            st.write(f"**Box {i+1}:** {label} ({conf:.1%})")

        st.divider()
        st.subheader("Final Labeled Image")
        st.image(final_draw_img, use_column_width=True)

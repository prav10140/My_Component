import os
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

st.set_page_config(page_title="Circuit Solver AI", page_icon="⚡")
st.title("⚡ Circuit Sketcher & Solver")

# --- SIDEBAR CALCULATOR ---
st.sidebar.header("🔢 Circuit Calculator (V=IR)")
calc_v = st.sidebar.number_input("Voltage (V)", value=0.0)
calc_i = st.sidebar.number_input("Current (I)", value=0.0)
calc_r = st.sidebar.number_input("Resistance (R)", value=0.0)

if st.sidebar.button("Solve Missing Value"):
    if calc_v > 0 and calc_r > 0:
        st.sidebar.success(f"Current (I) = {calc_v / calc_r:.2f} A")
    elif calc_i > 0 and calc_r > 0:
        st.sidebar.success(f"Voltage (V) = {calc_i * calc_r:.2f} V")
    elif calc_v > 0 and calc_i > 0:
        st.sidebar.success(f"Resistance (R) = {calc_v / calc_i:.2f} Ω")
    else:
        st.sidebar.warning("Enter at least two values!")

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
except Exception:
    st.error("AI Model failed to load.")
    st.stop()

LABELS = ['Ammeter', 'ac_src', 'battery', 'cap', 'curr_src', 'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src', 'dep_volt', 'diode', 'gnd_1', 'gnd_2', 'inductor', 'resistor', 'voltmeter']

# --- MAIN INTERFACE ---
mode = st.radio("Tool", ("freedraw", "rect"), horizontal=True)

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.1)", 
    stroke_width=2, # Keep lines sharp and thin
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=500,
    width=800,
    drawing_mode=mode,
    key="solver_board",
)

if canvas_result.json_data:
    objects = canvas_result.json_data.get("objects", [])
    rects = [obj for obj in objects if obj['type'] == 'rect']
    
    if rects and st.button("🔍 Recognize & Solve"):
        # 1. CLEAN IMAGE FOR MODEL (No UI lines)
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        sketch_only = Image.new("RGB", raw_img.size, (255, 255, 255))
        sketch_only.paste(raw_img, mask=raw_img.split()[3]) 
        
        # 2. SHARPEN (White lines on Black)
        full_gray = np.array(sketch_only.convert("L"))
        sharp_bw = cv2.adaptiveThreshold(full_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        sharp_pil = Image.fromarray(sharp_bw)

        labeled_img = sketch_only.copy()
        draw = ImageDraw.Draw(labeled_img)
        
        st.subheader("Recognized Components")
        cols = st.columns(min(len(rects), 4))

        for i, rect in enumerate(rects):
            l, t, w, h = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
            
            # Predict from cleaned sharp layer
            crop = sharp_pil.crop((l, t, l+w, t+h))
            input_arr = np.array(crop.resize((128, 128)).convert("RGB")) / 255.0
            preds = model.predict(np.expand_dims(input_arr, axis=0))
            
            label = LABELS[np.argmax(preds)]
            
            # 3. Labeling with UI exclusion
            draw.rectangle([l, t, l+w, t+h], outline="red", width=3)
            draw.text((l + 5, t + 5), f"{label}", fill="red")
            
            with cols[i % 4]:
                st.image(crop, caption=f"Box {i+1}", use_column_width=True)
                st.write(f"**{label}**")

        st.image(labeled_img, caption="Complete Annotated Circuit", use_column_width=True)

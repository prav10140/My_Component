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

st.set_page_config(page_title="Circuit Solver AI", page_icon="⚡", layout="wide")
st.title("⚡ Sharp Thread-Like Circuit Solver")
st.write("Sketch in **freedraw** mode. Use **Ctrl+Z** to undo. Switch to **rect** to box components for AI.")

# --- SIDEBAR CALCULATOR ---
st.sidebar.header("🔢 Ohm's Law Solver")
v_val = st.sidebar.number_input("Voltage (V)", value=0.0)
i_val = st.sidebar.number_input("Current (I)", value=0.0)
r_val = st.sidebar.number_input("Resistance (R)", value=0.0)

if st.sidebar.button("Solve"):
    if v_val > 0 and r_val > 0: st.sidebar.success(f"I = {v_val / r_val:.2f} A")
    elif i_val > 0 and r_val > 0: st.sidebar.success(f"V = {i_val * r_val:.2f} V")
    elif v_val > 0 and i_val > 0: st.sidebar.success(f"R = {v_val / i_val:.2f} Ω")
    else: st.sidebar.warning("Provide two values!")

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

# --- MAIN CANVAS ---
col_main, col_res = st.columns([2, 1])

with col_main:
    mode = st.radio("Tool", ("freedraw", "rect"), horizontal=True)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.1)", 
        stroke_width=2, # Thin lines like thread
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=500,
        width=750,
        drawing_mode=mode,
        display_toolbar=True, # Supports Undo/Redo & Shortcuts
        key="sharp_thread_canvas",
    )

if canvas_result.json_data:
    objects = canvas_result.json_data.get("objects", [])
    rects = [obj for obj in objects if obj['type'] == 'rect']
    
    if rects and st.button("🔍 Analyze Selection"):
        # 1. ISOLATE CLEAN SKETCH (Model ignores selection boxes)
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        sketch_only = Image.new("RGB", raw_img.size, (255, 255, 255))
        sketch_only.paste(raw_img, mask=raw_img.split()[3]) 
        
        # 2. SHARPENING: Binary Inversion (White lines on Black)
        # We use a specific threshold to keep thin lines sharp
        full_gray = np.array(sketch_only.convert("L"))
        # Constant 'C' adjusted to 5 to make thread-like lines pop against the black
        sharp_bw = cv2.adaptiveThreshold(full_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
        sharp_pil = Image.fromarray(sharp_bw)

        labeled_img = sketch_only.copy()
        draw = ImageDraw.Draw(labeled_img)
        
        with col_res:
            st.subheader("AI Analysis")
            for i, rect in enumerate(rects):
                l, t, w, h = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
                
                # Crop from the sharpened thread-like layer
                crop = sharp_pil.crop((l, t, l+w, t+h))
                
                # Preprocess for model (128x128 RGB)
                input_arr = np.array(crop.resize((128, 128)).convert("RGB")) / 255.0
                preds = model.predict(np.expand_dims(input_arr, axis=0))
                label = LABELS[np.argmax(preds)]
                
                st.image(crop, caption=f"Component {i+1}: {label}", width=150)
                
                # Draw on the original view for user display
                draw.rectangle([l, t, l+w, t+h], outline="red", width=3)
                draw.text((l + 5, t + 5), f"{label}", fill="red")

        st.divider()
        st.subheader("Annotated Whiteboard")
        st.image(labeled_img, use_column_width=True)

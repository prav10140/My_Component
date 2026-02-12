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
st.title("⚡ Sharp Multi-Mode Circuit Solver")

# --- SIDEBAR: OHM'S LAW CALCULATOR ---
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

# --- INPUT SELECTION ---
input_mode = st.radio("Choose Input Type", ("Whiteboard Sketch", "Upload & Edit Photo", "Live Camera"), horizontal=True)

final_base_image = None

if input_mode == "Whiteboard Sketch":
    col_main, col_res = st.columns([2, 1])
    with col_main:
        tool_mode = st.radio("Tool", ("freedraw", "rect"), horizontal=True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)", 
            stroke_width=2, 
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=500,
            width=750,
            drawing_mode=tool_mode,
            display_toolbar=True, # Support Ctrl+Z / Ctrl+Y
            key="sharp_thread_canvas",
        )
    if canvas_result.image_data is not None:
        raw_img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        final_base_image = Image.new("RGB", raw_img.size, (255, 255, 255))
        final_base_image.paste(raw_img, mask=raw_img.split()[3])

elif input_mode == "Upload & Edit Photo":
    uploaded_file = st.file_uploader("Upload a circuit photo", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        
        # Simple Rotation Addition
        angle = st.slider("Rotate Image", -180, 180, 0)
        if angle != 0:
            img = img.rotate(angle, expand=True)
        
        st.write("Now use 'Whiteboard Sketch' mode to draw a selection 'rect' over this image if you need to identify specific parts.")
        final_base_image = img
        st.image(img, caption="Uploaded Photo", width=500)

else:
    cam_file = st.camera_input("Take a photo of your circuit")
    if cam_file:
        final_base_image = Image.open(cam_file).convert("RGB")

# --- PROCESSING ---
if final_base_image and st.button("🔍 Analyze Circuit"):
    # Convert to sharp B&W thread-like structure
    full_gray = np.array(final_base_image.convert("L"))
    # Adaptive threshold ensures lines are bright and sharp
    sharp_bw = cv2.adaptiveThreshold(full_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    sharp_pil = Image.fromarray(sharp_bw)

    # Individual Component Analysis (using rectangles from Whiteboard if present)
    if input_mode == "Whiteboard Sketch" and canvas_result.json_data:
        objects = canvas_result.json_data.get("objects", [])
        rects = [obj for obj in objects if obj['type'] == 'rect']
        
        if rects:
            labeled_img = final_base_image.copy()
            draw = ImageDraw.Draw(labeled_img)
            
            for i, rect in enumerate(rects):
                l, t, w, h = int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height'])
                crop = sharp_pil.crop((l, t, l+w, t+h))
                
                # Predict
                prep = np.array(crop.resize((128, 128)).convert("RGB")) / 255.0
                preds = model.predict(np.expand_dims(prep, axis=0))
                label = LABELS[np.argmax(preds)]
                
                draw.rectangle([l, t, l+w, t+h], outline="red", width=3)
                draw.text((l + 5, t + 5), f"{label}", fill="red")
                st.write(f"**Component {i+1}:** {label}")
            
            st.image(labeled_img, caption="Analyzed Circuit", use_column_width=True)
        else:
            st.warning("Draw 'rect' boxes on the Whiteboard Sketch to identify specific components.")
    else:
        # Full Image Analysis for Uploaded/Camera photos
        st.subheader("Sharp AI View")
        st.image(sharp_pil, caption="Processed Image (AI Input)", width=400)
        
        prep = np.array(sharp_pil.resize((128, 128)).convert("RGB")) / 255.0
        preds = model.predict(np.expand_dims(prep, axis=0))
        st.success(f"**Full Image Prediction:** {LABELS[np.argmax(preds)]}")

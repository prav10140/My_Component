import os
import io
import base64
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Circuit Component Annotator", page_icon="🔌")
st.title("🔌 Circuit Component Annotator")
st.write("Draw a rectangle around a component, rotate/preview, predict, and stamp labels.")

# ---------------------------------------------------------
# IMPORT CANVAS
# ---------------------------------------------------------
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas-fix in requirements.txt")
    st.stop()

# ---------------------------------------------------------
# SETTINGS & MODEL LOADING
# ---------------------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"  # Replace with your File ID
DISPLAY_MAX_WIDTH = 800
INPUT_SIZE = (128, 128)

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

try:
    with st.spinner("Loading Model..."):
        model = load_model_from_drive()
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# ---------------------------------------------------------
# HELPER: BBOX EXTRACTION
# ---------------------------------------------------------
def extract_bbox(obj, scale, orig_w, orig_h):
    if not obj: return None
    l = int(obj['left'] * scale)
    t = int(obj['top'] * scale)
    w = int(obj['width'] * scale)
    h = int(obj['height'] * scale)
    return (max(0, l), max(0, t), min(orig_w, l+w), min(orig_h, t+h))

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Circuit Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1. Load and Resize Image
    orig_img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    # Calculate display scale
    display_w = min(DISPLAY_MAX_WIDTH, orig_w)
    scale_factor = orig_w / display_w
    display_h = int(orig_h / scale_factor)
    
    # Resize for the canvas (this is what the user sees)
    disp_img = orig_img.resize((display_w, display_h))

    # 2. THE CANVAS (FIXED)
    # We use 'background_image' directly with the PIL object.
    # We set 'fill_color' to transparent so the box isn't black.
    st.caption("Draw a box around a component:")
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",  # Fully transparent fill
        stroke_width=2,
        stroke_color="#FF0000",               # Red border
        background_image=disp_img,            # Direct PIL image (Fixes black background)
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="rect",
        key="canvas",
    )

    # 3. CROP & PREDICT LOGIC
    if st.button("Crop & Predict"):
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if objects:
                # Get the last drawn object
                obj = objects[-1]
                bbox = extract_bbox(obj, scale_factor, orig_w, orig_h)
                left, top, right, bottom = bbox
                
                # Validation
                if right - left < 5 or bottom - top < 5:
                    st.warning("Selection too small. Please draw a larger box.")
                else:
                    # Crop from ORIGINAL high-res image
                    crop = orig_img.crop(bbox)
                    st.session_state["last_crop"] = (crop, bbox)
                    st.rerun() # Refresh to show the rotation tool
            else:
                st.warning("Please draw a box first.")

    # 4. ROTATION & ACCEPTANCE UI
    if "last_crop" in st.session_state:
        st.divider()
        st.subheader("🔎 Verify & Stamp")
        crop, bbox = st.session_state["last_crop"]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            angle = st.slider("Rotate Component", -180, 180, 0)
            rotated_crop = crop.rotate(angle, expand=True)
            st.image(rotated_crop, caption="Preview", width=150)
        
        with col2:
            # Predict
            arr = np.array(rotated_crop.convert('L').resize(INPUT_SIZE))
            arr = np.expand_dims(np.stack([arr]*3, axis=-1) / 255.0, axis=0)
            
            preds = model.predict(arr)
            idx = np.argmax(preds)
            label = class_labels[idx]
            conf = float(np.max(preds))
            
            st.metric("Prediction", label, f"{conf:.1%}")
            
            if st.button("✅ Accept & Stamp"):
                if "annotations" not in st.session_state:
                    st.session_state["annotations"] = []
                st.session_state["annotations"].append({
                    "bbox": bbox, "label": label, "conf": conf
                })
                del st.session_state["last_crop"]
                st.rerun()

    # 5. DRAW ANNOTATIONS ON FINAL IMAGE
    if "annotations" in st.session_state and st.session_state["annotations"]:
        annotated_img = orig_img.copy()
        draw = ImageDraw.Draw(annotated_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for ann in st.session_state["annotations"]:
            l, t, r, b = ann["bbox"]
            draw.rectangle([l, t, r, b], outline="red", width=4)
            draw.text((l, t-15), f"{ann['label']} ({ann['conf']:.2f})", fill="red", font=font)
        
        st.divider()
        st.subheader("📝 Annotated Result")
        st.image(annotated_img, use_column_width=True)

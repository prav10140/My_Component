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
st.set_page_config(page_title="Circuit Component Annotator (ROBUST)", page_icon="🔌")
st.title("🔌 Circuit Component Annotator — ROBUST")
st.write("Draw a rectangle around a component, rotate/preview, predict, and stamp labels back on the circuit image.")

# ---------------------------------------------------------
# IMPORT CANVAS (With Error Handling)
# ---------------------------------------------------------
_has_canvas = True
try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    _has_canvas = False

# ---------------------------------------------------------
# SETTINGS & CONSTANTS
# ---------------------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"  # Google Drive File ID
DISPLAY_MAX_WIDTH = 900
INPUT_SIZE = (128, 128)

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# ---------------------------------------------------------
# MODEL DOWNLOAD & LOADER
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive…")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded.")

@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def pil_image_to_data_uri(pil_img, fmt="PNG"):
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def extract_bbox_from_obj(obj, scale, orig_w, orig_h):
    """
    Given a single canvas object dict, return a clipped (L,T,R,B) tuple or None.
    Defensive: handles missing keys, string values, non-rect objects, zero-area boxes.
    """
    try:
        obj_type = str(obj.get("type", "")).lower()
        if obj_type and obj_type not in ("rect", "rectangle", ""):
            return None

        left_f   = obj.get("left", 0)
        top_f    = obj.get("top", 0)
        width_f  = obj.get("width", obj.get("scaleX", 0) or 0)
        height_f = obj.get("height", obj.get("scaleY", 0) or 0)

        left   = int(round(float(left_f) * scale))
        top    = int(round(float(top_f)  * scale))
        width  = int(round(float(width_f) * scale))
        height = int(round(float(height_f)* scale))

        right  = left + width
        bottom = top  + height

        left   = max(0, left)
        top    = max(0, top)
        right  = min(orig_w, right)
        bottom = min(orig_h, bottom)

        if right <= left or bottom <= top:
            return None

        return (left, top, right, bottom)
    except Exception:
        return None

# ---------------------------------------------------------
# UI: IMAGE UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload a full circuit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if not uploaded_file:
    st.info("Upload a circuit image to begin.")
    st.stop()

orig_img = Image.open(uploaded_file).convert("RGB")
orig_w, orig_h = orig_img.size

# Scale for display
display_w = min(DISPLAY_MAX_WIDTH, orig_w)
scale = orig_w / display_w if display_w else 1.0
display_h = int(orig_h / scale) if scale else orig_h
disp_img = orig_img.resize((display_w, display_h))

st.caption("Draw a rectangle around ONE component, then click 'Crop & Predict'. Use the rotation slider before accepting.")

# ---------------------------------------------------------
# CREATE CANVAS
# ---------------------------------------------------------
canvas_result = None
if not _has_canvas:
    st.error("The 'streamlit-drawable-canvas' package is missing.")
    st.stop()

data_uri = pil_image_to_data_uri(disp_img, fmt="PNG")
tried_background_image_url = False

try:
    canvas_result = st_canvas(
        background_image_url=data_uri,
        stroke_width=2,
        stroke_color="#FF0000",
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="rect",
        key="component_canvas_robust",
    )
except TypeError:
    tried_background_image_url = True
except Exception:
    tried_background_image_url = True

if canvas_result is None and tried_background_image_url:
    try:
        canvas_result = st_canvas(
            background_image=disp_img,
            stroke_width=2,
            stroke_color="#FF0000",
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key="component_canvas_robust_b",
        )
    except Exception as e:
        st.error(f"Canvas failed to load: {e}")
        st.stop()

# ---------------------------------------------------------
# CROP & PREDICT UI
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Crop & Predict"):
        if canvas_result is None:
            st.error("Canvas not created.")
        else:
            objs = []
            try:
                objs = canvas_result.json_data.get("objects", []) if hasattr(canvas_result, "json_data") else []
            except Exception:
                objs = []

            if not objs:
                st.warning("No objects found. Draw a rectangle first.")
            else:
                obj = objs[-1]
                bbox = extract_bbox_from_obj(obj, scale, orig_w, orig_h)
                if bbox is None:
                    st.error("Invalid selection area.")
                else:
                    left, top, right, bottom = bbox
                    try:
                        crop = orig_img.crop((left, top, right, bottom))
                        st.session_state["last_crop"] = (crop, (left, top, right, bottom))
                        st.success(f"Crop ready.")
                    except Exception as e:
                        st.error(f"Error cropping: {e}")

with col2:
    if st.button("Clear last rectangle"):
        st.rerun()
with col3:
    if st.button("Clear all annotations"):
        st.session_state["annotations"] = []
        st.rerun()

# ---------------------------------------------------------
# ROTATION & ACCEPTANCE
# ---------------------------------------------------------
if "last_crop" in st.session_state:
    crop_img, bbox = st.session_state["last_crop"]
    st.subheader("Preview & Predict")
    colA, colB = st.columns([1,1])
    
    with colA:
        angle = st.slider("Rotate crop (degrees)", -180, 180, 0)
        rotated = crop_img.rotate(angle, expand=True)
        st.image(rotated.resize((256,256)), caption="Rotated crop preview")

        with st.spinner("Predicting…"):
            try:
                arr = np.array(rotated.convert('L'))
                norm = arr / 255.0
                if np.mean(norm) > 0.5:
                    arr = 255 - arr
                resized = cv2.resize(arr, INPUT_SIZE)
                stacked = np.stack([resized]*3, axis=-1)
                inp = np.expand_dims(stacked/255.0, axis=0)
                preds = model.predict(inp)
                pred_idx = int(np.argmax(preds))
                label = class_labels[pred_idx]
                conf = float(np.max(preds))
            except Exception as e:
                st.error(f"Prediction error: {e}")
                label, conf = "ERROR", 0.0

        if st.button(f"Accept: {label} ({conf:.2f})"):
            if "annotations" not in st.session_state:
                st.session_state["annotations"] = []
            st.session_state["annotations"].append({
                "bbox": bbox, "label": label, "confidence": conf
            })
            del st.session_state["last_crop"]
            st.success(f"Added: {label}")

    with colB:
        st.write(f"**Predicted Label:** {label}")
        st.write(f"**Confidence:** {conf:.2f}")

# ---------------------------------------------------------
# FINAL ANNOTATED IMAGE
# ---------------------------------------------------------
base = orig_img.copy()
draw = ImageDraw.Draw(base)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

if "annotations" in st.session_state and st.session_state["annotations"]:
    for ann in st.session_state["annotations"]:
        l, t, r, b = ann["bbox"]
        draw.rectangle([l, t, r, b], outline=(255,0,0), width=3)
        text = f"{ann['label']} ({ann['confidence']:.2f})"
        
        # Draw text background
        if hasattr(draw, "textbbox"): # New Pillow
            left, top, right, bottom = draw.textbbox((l, t), text, font=font)
            w, h = right - left, bottom - top
        else: # Old Pillow
            w, h = draw.textsize(text, font=font)
            
        draw.rectangle([l, t-h-6, l+w+6, t], fill=(255,0,0))
        draw.text((l+3, t-h-4), text, fill=(255,255,255), font=font)

st.subheader("Annotated Circuit")
st.image(base, use_column_width=True)

buf = io.BytesIO()
base.save(buf, format="PNG")
st.download_button("Download Image", data=buf.getvalue(), file_name="annotated_circuit.png", mime="image/png")

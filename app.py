# streamlit_circuit_annotator.py
# Recommended requirements (put in requirements.txt):
# streamlit
# tensorflow
# opencv-python-headless
# numpy
# pillow
# gdown
# streamlit-drawable-canvas-fix   # recommended to avoid Streamlit-internals issues
#
# If you must use the original package and an older Streamlit, pin:
# streamlit==1.24.1
# streamlit-drawable-canvas

import os
import io
import base64
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Circuit Component Annotator", page_icon="🔌")
st.title("🔌 Circuit Component Annotator")
st.write(
    "Upload a full circuit image, draw one component bounding box at a time, rotate/preview the crop, "
    "run the CNN classifier, and stamp the label back onto the main circuit image."
)

# ---------- SETTINGS ----------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"
DISPLAY_MAX_WIDTH = 900  # canvas display width (keeps UI manageable)
INPUT_SIZE = (128, 128)

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# ---------- MODEL DOWNLOAD & LOAD ----------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive…")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("Model downloaded")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# ---------- Helper: make data URI ----------
def pil_image_to_data_uri(pil_img, fmt="PNG"):
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

# ---------- UI: upload image ----------
uploaded_file = st.file_uploader("Upload a full circuit image (PNG/JPG)", type=["png", "jpg", "jpeg"]) 

if uploaded_file:
    orig_img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = orig_img.size

    # scale for display
    display_w = min(DISPLAY_MAX_WIDTH, orig_w)
    scale = orig_w / display_w
    display_h = int(orig_h / scale)
    disp_img = orig_img.resize((display_w, display_h))

    st.caption("Draw a rectangle around ONE component, then click 'Crop & Predict'. Repeat for each component.")

    # Create data URI for background (bypass internal st_image helper)
    data_uri = pil_image_to_data_uri(disp_img, fmt="PNG")

    # Try using background_image_url first (this avoids the Streamlit internals helper)
    canvas_result = None
    tried_background_image = False
    try:
        canvas_result = st_canvas(
            background_image_url=data_uri,
            stroke_width=2,
            stroke_color="#FF0000",
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key="component_canvas",
        )
    except TypeError as e:
        # older/newer canvas versions might not accept background_image_url argument
        st.warning("Canvas doesn't accept background_image_url — trying background_image fallback...")
        tried_background_image = True
    except Exception as e:
        # if a different AttributeError arises, we'll try fallback below
        st.warning("Canvas call with background_image_url failed, attempting fallback...")

    # Fallback: try passing the PIL image directly (this is the original approach that may fail on some Streamlit versions)
    if canvas_result is None and tried_background_image:
        try:
            canvas_result = st_canvas(
                background_image=disp_img,
                stroke_width=2,
                stroke_color="#FF0000",
                update_streamlit=True,
                height=display_h,
                width=display_w,
                drawing_mode="rect",
                key="component_canvas",
            )
        except Exception as e:
            # Both strategies failed, instruct the user about the fix
            st.error(
                "streamlit-drawable-canvas is incompatible with your Streamlit version in this environment.\n\n"
                "Two options to fix this:\n"
                "1) Install the community-fixed package: `pip install streamlit-drawable-canvas-fix` (recommended).\n"
                "2) Pin Streamlit to an older compatible version and use the original package:\n"
                "   `pip install streamlit==1.24.1 streamlit-drawable-canvas`\n\n"
                "After installing, restart the app."
            )
            st.stop()

    # If canvas_result remains None (rare), stop
    if canvas_result is None:
        st.error("Unable to create drawing canvas. Try installing 'streamlit-drawable-canvas-fix' or pin Streamlit.")
        st.stop()

    # Buttons to manage annotations
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Crop & Predict"):
            if canvas_result.json_data and "objects" in canvas_result.json_data and len(canvas_result.json_data["objects"])>0:
                obj = canvas_result.json_data["objects"][-1]  # take the most-recent rectangle
                left = int(obj["left"] * scale)
                top = int(obj["top"] * scale)
                width = int(obj["width"] * scale)
                height = int(obj["height"] * scale)

                # ensure within image
                left = max(0, left)
                top = max(0, top)
                right = min(orig_w, left + width)
                bottom = min(orig_h, top + height)

                if right - left <= 0 or bottom - top <= 0:
                    st.error("Invalid bounding box — try again.")
                else:
                    crop = orig_img.crop((left, top, right, bottom))
                    st.session_state["last_crop"] = (crop, (left, top, right, bottom))
            else:
                st.warning("Draw a rectangle first.")

    with col2:
        if st.button("Clear last rectangle"):
            # easiest way to clear canvas is to rerun; instruct user
            st.experimental_rerun()

    with col3:
        if st.button("Clear all annotations"):
            st.session_state["annotations"] = [] if "annotations" in st.session_state else []
            st.experimental_rerun()

    # Rotation & predict controls (operate on last crop)
    if "last_crop" in st.session_state:
        crop_img, bbox = st.session_state["last_crop"]
        st.subheader("Preview & Predict")
        colA, colB = st.columns([1,1])
        with colA:
            angle = st.slider("Rotate crop (degrees)", -180, 180, 0)
            rotated = crop_img.rotate(angle, expand=True)
            st.image(rotated.resize((256,256)), caption="Rotated crop preview", use_column_width=False)

            # prepare model input
            with st.spinner("Preparing & predicting…"):
                # convert to grayscale like original pipeline
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

            if st.button(f"Accept and stamp as: {label} ({conf:.2f})"):
                # add annotation to session state
                if "annotations" not in st.session_state:
                    st.session_state["annotations"] = []
                st.session_state["annotations"].append({
                    "bbox": bbox,
                    "label": label,
                    "confidence": conf
                })
                # remove last_crop so user draws next
                del st.session_state["last_crop"]
                st.success(f"Annotated: {label}")

        with colB:
            st.write("\n")
            st.write("[Model prediction details shown above]")

    # Build & display annotated image
    base = orig_img.copy()
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    if "annotations" in st.session_state and len(st.session_state["annotations"])>0:
        for ann in st.session_state["annotations"]:
            l,t,r,b = ann["bbox"]
            draw.rectangle([l,t,r,b], outline=(255,0,0), width=3)
            text = f"{ann['label']} ({ann['confidence']:.2f})"
            text_w, text_h = draw.textsize(text, font=font)
            draw.rectangle([l, t-text_h-6, l+text_w+6, t], fill=(255,0,0))
            draw.text((l+3, t-text_h-4), text, fill=(255,255,255), font=font)

    st.subheader("Annotated circuit")
    st.image(base, use_column_width=True)

    # download annotated image
    buf = io.BytesIO()
    base.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download annotated image (PNG)", data=byte_im, file_name="annotated_circuit.png", mime="image/png")

else:
    st.info("Upload a circuit image to begin.")

import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ==========================================================
# 🔧 STREAMLIT DRAWABLE CANVAS STABILITY PATCH
# ==========================================================
import streamlit.elements.image as st_image

if not hasattr(st_image, "image_to_url"):
    try:
        from streamlit.runtime.media_file_storage import media_file_storage
        from streamlit.elements.image import _normalize_to_bytes

        def image_to_url(image, width, clamp):
            image_bytes, mimetype = _normalize_to_bytes(image, width, clamp)
            file_id = media_file_storage.add(image_bytes, mimetype)
            return media_file_storage.get_url(file_id)

        st_image.image_to_url = image_to_url
    except Exception:
        pass

# Import AFTER patch
from streamlit_drawable_canvas import st_canvas

# ----------------------------------------------------------
# PAGE SETUP
# ----------------------------------------------------------
st.set_page_config(page_title="Circuit Annotator", page_icon="🔌")
st.title("🔌 Circuit Component Annotator")

# ----------------------------------------------------------
# MODEL CONFIG
# ----------------------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"

LABELS = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


try:
    with st.spinner("Loading AI model..."):
        model = load_model()
except Exception as e:
    st.error("❌ Model failed to load")
    st.stop()

# ----------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Circuit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    # ----------------------------------
    # LOAD + FIX IMAGE
    # ----------------------------------
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    orig_w, orig_h = image.size

    # ----------------------------------
    # DISPLAY RESIZE
    # ----------------------------------
    DISPLAY_WIDTH = 700
    scale_factor = orig_w / DISPLAY_WIDTH
    display_h = int(orig_h / scale_factor)

    disp_img = image.resize(
        (DISPLAY_WIDTH, display_h)
    ).convert("RGBA")

    st.write("✏️ Draw rectangle around a component")

    # ----------------------------------
    # CANVAS
    # ----------------------------------
    canvas_result = st_canvas(
        fill_color="rgba(255,165,0,0.1)",
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=disp_img,
        update_streamlit=True,
        height=display_h,
        width=DISPLAY_WIDTH,
        drawing_mode="rect",
        key="canvas",
    )

    # ----------------------------------
    # PROCESS DRAWING
    # ----------------------------------
    if canvas_result.json_data is not None:

        objects = canvas_result.json_data.get("objects", [])

        if objects:
            obj = objects[-1]

            left = int(obj["left"] * scale_factor)
            top = int(obj["top"] * scale_factor)
            width = int(obj["width"] * scale_factor)
            height = int(obj["height"] * scale_factor)

            left = max(0, left)
            top = max(0, top)
            right = min(orig_w, left + width)
            bottom = min(orig_h, top + height)

            if st.button("🔍 Crop & Analyze"):

                if width < 5 or height < 5:
                    st.warning("Selection too small")
                    st.stop()

                crop = image.crop((left, top, right, bottom))
                st.image(crop, caption="Cropped Component", width=160)

                # ----------------------------------
                # PREDICTION
                # ----------------------------------
                try:
                    resized = crop.resize((128, 128))
                    arr = np.array(resized) / 255.0
                    arr = np.expand_dims(arr, axis=0)

                    preds = model.predict(arr)
                    idx = np.argmax(preds)
                    confidence = np.max(preds)

                    st.success(
                        f"### 🧠 Prediction: {LABELS[idx]}"
                    )
                    st.info(
                        f"Confidence: {confidence:.2%}"
                    )

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

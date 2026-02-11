import os
import io
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ==========================================
# 🔧 CRITICAL FIX FOR STREAMLIT 1.30+
# ==========================================
# This block fixes the "AttributeError: image_to_url" crash
import streamlit.elements.image as st_image
try:
    if not hasattr(st_image, 'image_to_url'):
        from streamlit.elements.utils import image_to_url
        st_image.image_to_url = image_to_url
except ImportError:
    pass  # If the patch fails, we proceed (it might work on some versions)
# ==========================================

# Import Canvas AFTER the fix
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas in requirements.txt")
    st.stop()

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Circuit Annotator", page_icon="🔌")
st.title("🔌 Circuit Annotator")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_" 

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    # compile=False ensures compatibility with newer TensorFlow versions
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    with st.spinner("Loading AI Model..."):
        model = load_model()
except Exception:
    st.warning("Model failed to download. Check the File ID.")

# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Circuit Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1. LOAD IMAGE
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image) # Fix phone rotation
    image = image.convert("RGB")           # Fix black screen transparency
    
    # 2. RESIZE FOR DISPLAY
    orig_w, orig_h = image.size
    DISPLAY_WIDTH = 700
    scale_factor = orig_w / DISPLAY_WIDTH
    display_h = int(orig_h / scale_factor)
    disp_img = image.resize((DISPLAY_WIDTH, display_h))

    st.write("Draw a box around a component:")

    # 3. CANVAS (Now patched to work on Streamlit Cloud)
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",  # Transparent fill
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=disp_img,            # PASSING PIL IMAGE
        update_streamlit=True,
        height=display_h,
        width=DISPLAY_WIDTH,
        drawing_mode="rect",
        key="canvas_patched",
    )

    # 4. CROP & PREDICT
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[-1]
            
            # Scale coordinates
            left = int(obj["left"] * scale_factor)
            top = int(obj["top"] * scale_factor)
            width = int(obj["width"] * scale_factor)
            height = int(obj["height"] * scale_factor)
            
            # Clip to image boundaries
            left = max(0, left)
            top = max(0, top)
            right = min(orig_w, left + width)
            bottom = min(orig_h, top + height)

            if st.button("Crop & Analyze"):
                if width > 5 and height > 5:
                    crop = image.crop((left, top, right, bottom))
                    st.image(crop, caption="Cropped Component", width=150)
                    
                    if 'model' in locals() and model:
                        try:
                            # Preprocess
                            resized = crop.resize((128, 128))
                            arr = np.array(resized) / 255.0
                            arr = np.expand_dims(arr, axis=0)
                            
                            # Predict
                            preds = model.predict(arr)
                            labels = [
                                'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
                                'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
                                'dep_volt', 'diode', 'gnd_1', 'gnd_2',
                                'inductor', 'resistor', 'voltmeter'
                            ]
                            idx = np.argmax(preds)
                            st.success(f"**{labels[idx]}** ({np.max(preds):.1%})")
                        except Exception as e:
                            st.error(f"Prediction Error: {e}")
                else:
                    st.warning("Selection too small!")

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
st.set_page_config(page_title="Circuit Annotator", page_icon="🔌")
st.title("🔌 Circuit Annotator")

# ---------------------------------------------------------
# IMPORT CANVAS
# ---------------------------------------------------------
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas-fix in requirements.txt")
    st.stop()

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"  # Replace if needed

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model_from_drive()
except Exception:
    st.warning("Model loading failed. Check File ID. (App will run without prediction)")

# ---------------------------------------------------------
# HELPER: FIXED IMAGE CONVERTER
# ---------------------------------------------------------
def get_image_data_url(pil_image):
    """
    FIX: Converts PIL image to a Base64 Data URL string.
    This prevents the 'Black Screen' issue on Streamlit Cloud.
    """
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Circuit Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1. Load Image & Force RGB (Fixes transparency blacking out)
    orig_img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    # 2. Resize for Display
    DISPLAY_WIDTH = 700
    scale_factor = orig_w / DISPLAY_WIDTH
    display_h = int(orig_h / scale_factor)
    disp_img = orig_img.resize((DISPLAY_WIDTH, display_h))

    # 3. CONVERT TO DATA URL (Crucial Step for Cloud)
    bg_image_url = get_image_data_url(disp_img)

    st.write("Draw a box around a component:")

    # 4. THE CANVAS
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",  # Transparent fill
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=None,                # Do NOT pass object
        background_color="#FFFFFF",           # White background fallback
        background_image_url=bg_image_url,    # Pass the Base64 String
        update_streamlit=True,
        height=display_h,
        width=DISPLAY_WIDTH,
        drawing_mode="rect",
        key="canvas_fixed",
    )

    # 5. CROP LOGIC
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[-1]
            
            # Calculate coordinates
            left = int(obj["left"] * scale_factor)
            top = int(obj["top"] * scale_factor)
            width = int(obj["width"] * scale_factor)
            height = int(obj["height"] * scale_factor)
            
            # Fix negative/out of bounds
            left = max(0, left)
            top = max(0, top)
            right = min(orig_w, left + width)
            bottom = min(orig_h, top + height)

            # Show Preview
            if st.button("Crop & Analyze"):
                if width > 5 and height > 5:
                    crop = orig_img.crop((left, top, right, bottom))
                    st.image(crop, caption="Cropped View", width=150)
                    
                    # Prediction Stub
                    try:
                        resized = crop.resize((128, 128))
                        arr = np.array(resized) / 255.0
                        arr = np.expand_dims(arr, axis=0)
                        preds = model.predict(arr)
                        st.success(f"Prediction: {np.argmax(preds)}")
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                else:
                    st.warning("Box too small!")

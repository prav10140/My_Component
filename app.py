import os
import io
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="Circuit Annotator", page_icon="🔌")
st.title("🔌 Circuit Annotator")

# ---------------------------------------------------------
# IMPORT CANVAS (Using the FIX library)
# ---------------------------------------------------------
try:
    # We use the 'fix' library to prevent crashes on new Streamlit versions
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas-fix in requirements.txt")
    st.stop()

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_"  # Replace with your File ID

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    # Using compile=False prevents errors with newer TF versions loading older models
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    with st.spinner("Loading Model..."):
        model = load_model_from_drive()
except Exception as e:
    st.warning(f"Model loading failed: {e}")

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Circuit Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1. LOAD & PREPARE IMAGE
    # convert("RGB") is CRITICAL. It removes the Alpha channel which causes the "Black Screen".
    orig_img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    # 2. RESIZE FOR DISPLAY
    DISPLAY_WIDTH = 700
    scale_factor = orig_w / DISPLAY_WIDTH
    display_h = int(orig_h / scale_factor)
    disp_img = orig_img.resize((DISPLAY_WIDTH, display_h))

    st.write("Draw a box around a component:")

    # 3. THE CANVAS
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",  # 0.0 opacity = Transparent (Fixes black box)
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=disp_img,            # Pass the PIL image directly
        update_streamlit=True,
        height=display_h,
        width=DISPLAY_WIDTH,
        drawing_mode="rect",
        key="canvas_modern",
    )

    # 4. CROP & PREDICT
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[-1]
            
            # Scale coordinates back to original image size
            left = int(obj["left"] * scale_factor)
            top = int(obj["top"] * scale_factor)
            width = int(obj["width"] * scale_factor)
            height = int(obj["height"] * scale_factor)
            
            # Cleanup coordinates
            left = max(0, left)
            top = max(0, top)
            right = min(orig_w, left + width)
            bottom = min(orig_h, top + height)

            if st.button("Crop & Analyze"):
                if width > 5 and height > 5:
                    crop = orig_img.crop((left, top, right, bottom))
                    st.image(crop, caption="Cropped Component", width=150)
                    
                    if 'model' in locals() and model:
                        try:
                            # Preprocess
                            resized = crop.resize((128, 128))
                            arr = np.array(resized) / 255.0
                            arr = np.expand_dims(arr, axis=0)
                            
                            # Predict
                            preds = model.predict(arr)
                            class_labels = [
                                'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
                                'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
                                'dep_volt', 'diode', 'gnd_1', 'gnd_2',
                                'inductor', 'resistor', 'voltmeter'
                            ]
                            idx = np.argmax(preds)
                            label = class_labels[idx]
                            conf = float(np.max(preds))
                            
                            st.success(f"**{label}** ({conf:.1%})")
                        except Exception as e:
                            st.error(f"Prediction Error: {e}")
                else:
                    st.warning("Box too small!")

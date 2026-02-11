import os
import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Circuit Annotator", page_icon="🔌")
st.title("🔌 Circuit Annotator")

# Import Canvas
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas in requirements.txt")
    st.stop()

# Load Model
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "1az8IY3x9E8jzePRz2QB3QjIhgGafjaH_" 

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    with st.spinner("Loading Model..."):
        model = load_model()
except Exception:
    st.warning("Model failed to load. Check File ID.")

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Circuit Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1. LOAD IMAGE
    image = Image.open(uploaded_file)
    
    # FIX A: Handle Rotation (Phone photos often look rotated without this)
    image = ImageOps.exif_transpose(image) 
    
    # FIX B: Force RGB (Removes transparency that turns black)
    image = image.convert("RGB")
    
    # 2. RESIZE
    orig_w, orig_h = image.size
    DISPLAY_WIDTH = 700
    scale_factor = orig_w / DISPLAY_WIDTH
    display_h = int(orig_h / scale_factor)
    disp_img = image.resize((DISPLAY_WIDTH, display_h))

    # FIX C: CONVERT TO NUMPY ARRAY (The "Black Screen" Killer)
    # Streamlit Cloud handles NumPy arrays much better than PIL objects
    img_array = np.array(disp_img)

    st.write("Draw a box around a component:")

    # 3. CANVAS
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",  # Transparent fill
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=img_array,           # <--- PASSING NUMPY ARRAY HERE
        update_streamlit=True,
        height=display_h,
        width=DISPLAY_WIDTH,
        drawing_mode="rect",
        key="canvas_numpy",
    )

    # 4. CROP & PREDICT
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[-1]
            
            # Scale & Clip
            left = int(obj["left"] * scale_factor)
            top = int(obj["top"] * scale_factor)
            width = int(obj["width"] * scale_factor)
            height = int(obj["height"] * scale_factor)
            
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
                            # Prepare for Model
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

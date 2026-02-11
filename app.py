import os
import io
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
    st.error("Please install streamlit-drawable-canvas in requirements.txt")
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

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Circuit Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1. LOAD IMAGE
    # We enforce RGB to prevent the "Black Screen" issue caused by RGBA transparency layers
    orig_img = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = orig_img.size
    
    # 2. RESIZE FOR DISPLAY
    DISPLAY_WIDTH = 700
    scale_factor = orig_w / DISPLAY_WIDTH
    display_h = int(orig_h / scale_factor)
    disp_img = orig_img.resize((DISPLAY_WIDTH, display_h))

    st.write("Draw a box around a component:")

    # 3. THE CANVAS
    # We pass the PIL image directly to background_image
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",  # Transparent fill (0.0 opacity)
        stroke_width=2,
        stroke_color="#FF0000",
        background_image=disp_img,            # Direct PIL Object
        update_streamlit=True,
        height=display_h,
        width=DISPLAY_WIDTH,
        drawing_mode="rect",
        key="canvas_fixed",
    )

    # 4. CROP & ANALYZE LOGIC
    if canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        if objects:
            obj = objects[-1]
            
            # Extract coordinates with scaling
            left = int(obj["left"] * scale_factor)
            top = int(obj["top"] * scale_factor)
            width = int(obj["width"] * scale_factor)
            height = int(obj["height"] * scale_factor)
            
            # Boundary checks
            left = max(0, left)
            top = max(0, top)
            right = min(orig_w, left + width)
            bottom = min(orig_h, top + height)

            if st.button("Crop & Analyze"):
                if width > 5 and height > 5:
                    # Crop from the high-res original image
                    crop = orig_img.crop((left, top, right, bottom))
                    st.image(crop, caption="Cropped Component", width=150)
                    
                    # Prediction
                    try:
                        # Preprocessing for Model
                        resized = crop.resize((128, 128))
                        arr = np.array(resized)
                        
                        # Normalize (match your training data)
                        arr = arr / 255.0
                        arr = np.expand_dims(arr, axis=0)
                        
                        preds = model.predict(arr)
                        idx = np.argmax(preds)
                        label = class_labels[idx]
                        conf = float(np.max(preds))
                        
                        st.success(f"**{label}** ({conf:.1%})")
                        
                        # Add to session state if you want to save it
                        if "annotations" not in st.session_state:
                            st.session_state["annotations"] = []
                        st.session_state["annotations"].append({
                            "bbox": (left, top, right, bottom),
                            "label": label,
                            "conf": conf
                        })
                        
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                else:
                    st.warning("Selection too small!")
    
    # 5. SHOW ALL ANNOTATIONS
    if "annotations" in st.session_state and st.session_state["annotations"]:
        st.divider()
        st.write("### Annotated Result")
        annotated_img = orig_img.copy()
        draw = ImageDraw.Draw(annotated_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for ann in st.session_state["annotations"]:
            l, t, r, b = ann["bbox"]
            draw.rectangle([l, t, r, b], outline="red", width=5)
            draw.text((l, t-20), f"{ann['label']}", fill="red", font=font)
        
        st.image(annotated_img, use_column_width=True)

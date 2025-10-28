import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model only once
model = load_model("MY_MODEL.keras")

class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # ✅ Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ✅ Normalize and invert if background is white
        norm = gray / 255.0
        if np.mean(norm) > 0.5:
            gray = 255 - gray

        # ✅ Resize & stack to 3 channels
        gray_resized = cv2.resize(gray, (128, 128))
        img_rgb = cv2.merge([gray_resized, gray_resized, gray_resized])
        img_input = np.expand_dims(img_rgb / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_input)
        pred_class = np.argmax(prediction)
        label = class_labels[pred_class]

        return jsonify({"label": label})
    except Exception as e:
        return jsonify({"error": str(e)})

# Vercel looks for "app"
if __name__ == "__main__":
    app.run()

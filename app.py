import os
import base64
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -------------------------------
# 1️⃣ Download model from Google Drive if not present
MODEL_PATH = "MY_MODEL.keras"
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("Model already exists locally.")

# Load model
model = load_model(MODEL_PATH)

# Class labels
class_labels = [
    'Ammeter', 'ac_src', 'battery', 'cap', 'curr_src',
    'dc_volt_src_1', 'dc_volt_src_2', 'dep_curr_src',
    'dep_volt', 'diode', 'gnd_1', 'gnd_2',
    'inductor', 'resistor', 'voltmeter'
]

# -------------------------------
# API endpoint
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize & invert if background is white
        norm = gray / 255.0
        if np.mean(norm) > 0.5:
            gray = 255 - gray

        # Resize & stack to 3 channels
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

# -------------------------------
# Frontend page
@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Live CNN Prediction</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"></script>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            text-align: center;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #fff;
            margin: 0;
            padding: 0;
        }
        h1 {
            margin-top: 20px;
            font-size: 2.5em;
            text-shadow: 2px 2px 8px #000;
        }
        #result {
            margin-top: 15px;
            font-size: 1.5em;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            display: inline-block;
            border-radius: 10px;
        }
        canvas {
            border: 3px solid #fff;
            margin-top: 20px;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <h1>📷 Live CNN Prediction</h1>
    <div id="result">Waiting...</div>

    <script>
        let video;
        const apiURL = "/api/predict";

        function setup() {
            createCanvas(128, 128);
            video = createCapture(VIDEO);
            video.size(128, 128);
            video.hide();
            frameRate(1);
        }

        async function draw() {
            image(video, 0, 0, 128, 128);
            const base64Img = document.querySelector("canvas").toDataURL("image/png");

            try {
                const res = await fetch(apiURL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image: base64Img })
                });
                const data = await res.json();
                document.getElementById("result").innerText = "🔮 " + (data.label || data.error);
            } catch (err) {
                document.getElementById("result").innerText = "Error: " + err;
            }
        }
    </script>
</body>
</html>
""")

# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

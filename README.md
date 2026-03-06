# Circuit Component Detector Model – Handwritten Circuit Component Detection

## Overview
Circuit Solver AI is an intelligent web application that detects **electronic components from hand-drawn circuit diagrams**. Users can sketch a circuit on a digital whiteboard or capture a photo using their camera, and the AI model will identify the components present in the circuit.

The system uses a **TensorFlow deep learning model** trained on circuit component images to classify components such as resistors, capacitors, diodes, and voltage sources.

The application also includes a **built-in Ohm’s Law calculator** to help students quickly compute voltage, current, or resistance.

This project aims to make **electronics learning more interactive** by allowing students to draw circuits and instantly understand the components.

---

# 🚀 Features

### Interactive Whiteboard
Users can draw circuits directly in the browser using a digital canvas.

### Live Camera Input
Users can capture a photo of a circuit diagram and analyze it instantly.

### AI Component Detection
The AI model detects electronic components from selected regions or the entire image.

### Supported Components
The model can identify:

- Ammeter  
- AC Source  
- Battery  
- Capacitor  
- Current Source  
- DC Voltage Source  
- Dependent Current Source  
- Dependent Voltage Source  
- Diode  
- Ground  
- Inductor  
- Resistor  
- Voltmeter  

### Ohm’s Law Calculator
Quickly solve:

- Voltage (V)
- Current (I)
- Resistance (R)



---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/circuit-solver-ai.git
cd circuit-solver-ai
```
## 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 3️⃣ Run the Application

```bash
streamlit run app.py
```


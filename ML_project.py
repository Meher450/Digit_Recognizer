# digit_recognizer_app.py

import gradio as gr
import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# ---------- Step 1: Train the model ----------
def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    joblib.dump(clf, "digit_classifier.pkl")
    print("Model trained and saved as 'digit_classifier.pkl'.")

# Train only if model doesn't exist
if not os.path.exists("digit_classifier.pkl"):
    train_model()

# ---------- Step 2: Define prediction function ----------
clf = joblib.load("digit_classifier.pkl")

def predict_digit(img):
    if img is None:
        return "Please draw a digit."

    # Convert image to grayscale and resize to 8x8
    img = img.convert('L').resize((8, 8), Image.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array / 16.0  # Scale 0–255 to 0–16
    img_array = 16 - img_array    # Invert colors to match training data
    img_array = img_array.flatten().reshape(1, -1)

    prediction = clf.predict(img_array)[0]
    return f"Predicted Digit: {prediction}"

# ---------- Step 3: Define Gradio Interface ----------
with gr.Blocks() as app:
    gr.Markdown("## 🧠 Digit Recognizer (8x8 Digits)")
    gr.Markdown("Draw a digit (0–9) below. The model will try to guess what it is.")

    with gr.Row():
        image_input = gr.Image(
            image_mode="L",
            label="Draw Here",
            interactive=True,
            width=200,
            height=200,
            type="pil"
        )
        output_text = gr.Textbox(label="Prediction")

    with gr.Row():
        predict_btn = gr.Button("Predict")
        clear_btn = gr.Button("Clear")

    # Events
    predict_btn.click(fn=predict_digit, inputs=image_input, outputs=output_text)
    clear_btn.click(fn=lambda: (None, ""), inputs=[], outputs=[image_input, output_text])

# ---------- Step 4: Launch ----------
if __name__ == "__main__":
    app.launch()
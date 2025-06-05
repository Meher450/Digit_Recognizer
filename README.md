# 🧠 Digit Recognizer App (8x8 Handwritten Digits)

A simple yet powerful machine learning project that recognizes handwritten digits using a **Random Forest classifier** and an interactive **Gradio interface**.

---

## 📌 Project Overview

This project demonstrates a classic digit classification use case using the `load_digits` dataset from `sklearn.datasets`. The app allows users to draw a digit (0–9), and the model predicts the number in real-time.

**Key Features:**
- Trained on 8x8 grayscale image data (scikit-learn digits dataset)
- Interactive UI built with Gradio
- Real-time digit prediction
- Lightweight and beginner-friendly
- 
---

## 🛠️ Tech Stack

- **Python 3.11**
- **scikit-learn**
- **Gradio**
- **PIL (Pillow)**
- **NumPy**
- **joblib**

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/digit-recognizer-app.git
cd digit-recognizer-app
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
python ML_project.py
```

This will open the Gradio interface in your browser.

---

## 🧪 How It Works

1. **Model Training**

   * Uses `RandomForestClassifier` from scikit-learn
   * Trained on the 8x8 pixel images of digits (0–9)

2. **Image Preprocessing**

   * User input is converted to an 8x8 grayscale image
   * Normalized and inverted to match training data

3. **Prediction**

   * The image is flattened and passed to the model for prediction
   * Result is displayed in real-time

---

## 📂 Project Structure

```
.
├── ML_project.py         # Main app script
├── digit_classifier.pkl  # Trained model (auto-generated)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📈 Future Improvements

* Add CNN model for better accuracy
* Include a training accuracy/evaluation section
* Deploy online with Hugging Face Spaces or Streamlit Cloud

---

## 🤝 Contributing

Feel free to fork the repo and submit pull requests. Feedback and improvements are welcome!

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).


---

Let me know if you'd like to customize the author section, add GIFs/screenshots, or generate a `requirements.txt`.
```

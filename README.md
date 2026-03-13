# 🧠 NeuraScan: Brain Tumor Classification AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-NeuraScan-blue?style=for-the-badge&logo=render)](https://neurascan-frontend.onrender.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.14%25-green?style=for-the-badge)](https://neurascan-frontend.onrender.com)
[![Python](https://img.shields.io/badge/Backend-Python%20%2F%20Flask-blue?style=flat-square&logo=python)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)

NeuraScan is a state-of-the-art medical imaging application designed to assist in the detection and classification of brain tumors from MRI scans. Using a high-performance **CNN + SVM Hybrid Architecture**, it provides accurate diagnostic insights along with visual explanations.

---

## 🚀 Live Link
Experience the AI in action: **[https://neurascan-frontend.onrender.com](https://neurascan-frontend.onrender.com)**

---

## ✨ Key Features
- **Accurate Classification**: Identifies 4 categories: Glioma, Meningioma, Pituitary, or No Tumor.
- **Hybrid AI Model**: Combines deep feature extraction via **EfficientNetB0** with the robust classification power of **SVM**.
- **Real-time Analysis**: Instant results with confidence scores and class probability breakdowns.
- **Medical Insights**: Provides descriptions, severity levels, and suggested treatment paths for each tumor type.
- **Grad-CAM Visualization**: Highlights the specific regions of the MRI scan that influenced the AI's decision.
- **Modern UI**: Clean, responsive dashboard built for medical professionals and researchers.

---

## 🛠️ Tech Stack
### **Backend**
- **Python / Flask**: Restful API for model serving.
- **TensorFlow & Keras**: Deep learning engine using EfficientNetB0.
- **Scikit-learn**: SVM implementation for hybrid classification.
- **OpenCV**: Image preprocessing and Grad-CAM generation.

### **Frontend**
- **React.js**: Modern, component-based user interface.
- **Axios**: Seamless communication with the backend.
- **CSS3**: Custom glassmorphism-inspired design with responsive layouts.

### **Deployment**
- **Render**: Scalable hosting for both API and static frontend.

---

## 🧠 Model Architecture & Performance
NeuraScan achieves an impressive **97.14% Accuracy** on testing datasets.

1. **Feature Extraction**: EfficientNetB0 (pretrained on ImageNet) extracts complex spatial features from MRI scans.
2. **Hybrid Prediction**: These features are fed into a dual pipeline:
   - **Softmax Layer**: Provides deep learning probabilities.
   - **Support Vector Machine (SVM)**: Provides high-margin classification.
3. **Decision Logic**: The system dynamically selects the most confident prediction to ensure maximum reliability.

---

## 💻 Local Setup

### **1. Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
*Port: 5000*

### **2. Frontend Setup**
```bash
cd frontend
npm install
npm start
```
*Port: 3000*

---

## 📂 Project Structure
```text
brain-tumor-app/
├── backend/            # Flask API & AI Logic
│   ├── app.py          # Main application server
│   ├── brain_tumor_97.keras # Pre-trained DL Model
│   ├── svm_model.pkl   # Hybrid SVM weights
│   └── requirements.txt
├── frontend/           # React Application
│   ├── src/
│   │   ├── App.js      # Primary UI Logic
│   │   └── App.css     # Styling
│   └── package.json
├── render.yaml         # Deployment Configuration
└── README.md           # Documentation
```

---

## ⚕️ Medical Disclaimer
**Warning:** This application is intended for **research and educational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.

---

## 👨‍💻 Author
**Vishnu**  
[GitHub Profile](https://github.com/Vishnu1124-v)

# 🧠 NeuraScan: Brain Tumor Classification AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-NeuraScan-blue?style=for-the-badge&logo=render)](https://neurascan-frontend.onrender.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.14%25-green?style=for-the-badge)](https://neurascan-frontend.onrender.com)
[![Python](https://img.shields.io/badge/Backend-Python%20%2F%20Flask-blue?style=flat-square&logo=python)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)

NeuraScan is a high-precision medical imaging application that leverages Deep Learning and Classical Machine Learning to classify brain tumors from MRI scans. Developed as a decision-support tool, it combines the representational power of **EfficientNetB0** with the robustness of **Support Vector Machines (SVM)**.

---

## 🚀 Live Access
**[View Web Application](https://neurascan-frontend.onrender.com)**  
> *Note: The application is hosted on a free tier. If it's your first time visiting, it may take ~1 minute to "wake up" the server (Cold Start).*

---

## 📊 Dataset Overview
The model was trained on the **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** (provided by Masoud Nickparvar on Kaggle).
- **Total Images**: 7,023 MRI scans.
- **Categories**:
  - 🔴 **Glioma**: Tumors originating in glial cells.
  - 🟠 **Meningioma**: Tumors forming in the membranes surrounding the brain.
  - 🔵 **Pituitary**: Tumors localized in the pituitary gland.
  - ✅ **No Tumor**: Healthy brain MRI scans.

---

## ⚙️ How It Works (The AI Pipeline)

NeuraScan uses a **Hybrid AI Architecture** to maximize accuracy and minimize false negatives.

### 1. Image Preprocessing
Every MRI scan undergoes a standardized pipeline:
- **Resizing**: Scaled to $224 \times 224$ pixels.
- **Normalization**: Pixel values adjusted to the EfficientNet range.
- **Contrast Enhancement**: Optimization for clear feature visibility.

### 2. Dual-Engine Prediction
The system doesn't rely on just one model. It uses a **Hybrid Pipeline**:
- **Deep Engine (EfficientNetB0)**: A state-of-the-art Convolutional Neural Network (CNN) pretrained on ImageNet. It acts as a powerful feature extractor, identifying complex edges, textures, and anomalies in the brain tissue.
- **Classical Engine (SVM)**: A Support Vector Machine trained on the "deep features" extracted by the CNN. SVM is highly effective at finding the optimal "boundary" between classes, providing a second layer of verification.

### 3. Explainability (Grad-CAM)
We believe medical AI should not be a "black box."
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** generates a heatmap over the original MRI.
- It highlights exactly which regions of the brain the AI focused on to reach its conclusion, allowing doctors to verify the findings visually.

---

## 🛠️ Tech Stack
| Layer | Technologies |
|---|---|
| **Frontend** | React.js, Axios, CSS3 (Glassmorphism) |
| **Backend** | Python, Flask, Gunicorn |
| **AI Framework** | TensorFlow, Keras, Scikit-learn |
| **CV Library** | OpenCV (Heatmap generation), Pillow |
| **Deployment** | Render (Web Services + Static Hosting) |

---

## 📈 Performance Metrics
- **Final Test Accuracy**: ~97.14%
- **Training Strategy**: 
  - **Phase 1**: Feature Extraction (Base model frozen).
  - **Phase 2**: Fine-Tuning (Top 30 layers of EfficientNet unfrozen).
- **Inference Time**: < 1 second (on warmed-up server).

---

## 📂 Project Structure
```text
brain-tumor-app/
├── backend/            # Flask API & AI Logic
│   ├── app.py          # Main application server (Prediction & Grad-CAM)
│   ├── brain_tumor_97.keras # Pre-trained Keras Model
│   ├── svm_model.pkl   # Hybrid SVM weights
│   ├── scaler.pkl      # Feature normalization weights
│   └── requirements.txt
├── frontend/           # React Application
│   ├── src/
│   │   ├── App.js      # Main Dashboard & API logic
│   │   └── App.css     # Premium UI styling
│   └── package.json
├── render.yaml         # Blueprint for Render Deployment
└── README.md           # This documentation
```

---

## 💻 Local Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/Vishnu1124-v/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### **2. Backend Setup**
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### **3. Frontend Setup**
```bash
cd ../frontend
npm install
npm start
```

---

## ⚕️ Medical Disclaimer
**Warning:** This application is for **research and educational purposes only**. It is not a clinical tool for medical diagnosis. The results should always be interpreted by a qualified medical professional.

---

## 👨‍💻 Author
**Vishnu**  
Professional AI/ML Developer & Student.  
[GitHub Profile](https://github.com/Vishnu1124-v)

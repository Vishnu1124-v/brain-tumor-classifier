from flask import Flask, request, jsonify
from flask_cors import CORS
import os
# Fix for Keras 3 serialization bug on Render
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import keras
import tensorflow as tf
import numpy as np
import pickle
import cv2
import base64
import os
from PIL import Image
import io
import gc

app = Flask(__name__)
# Explicitly allow the frontend URL to prevent any CORS blocks
CORS(app, resources={r"/*": {"origins": ["https://neurascan-frontend.onrender.com", "http://localhost:3000"]}})

# ── Load model & artifacts ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables for models (loaded lazily)
model = None
svm = None
scaler = None
CLASS_NAMES = None
prediction_model = None
grad_model = None

def load_models_lazy():
    global model, svm, scaler, CLASS_NAMES, prediction_model, grad_model
    if model is not None:
        return
        
    print("🔄 Loading models (Lazy Init)...")
    try:
        model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'brain_tumor_97.keras'))
        
        with open(os.path.join(BASE_DIR, 'svm_model.pkl'), 'rb') as f: 
            svm = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f: 
            scaler = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'class_names.pkl'), 'rb') as f: 
            CLASS_NAMES = pickle.load(f)
            
        prediction_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.layers[-4].output, model.output]
        )
        
        # Pre-initialize Grad-CAM model (DISABLED in Slim Mode to save RAM)
        # grad_model = tf.keras.Model(
        #     inputs=model.inputs,
        #     outputs=[model.get_layer('top_conv').output, model.output]
        # )
        print("✅ All artifacts loaded successfully!")
        gc.collect() # Force cleanup immediately after loading
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise e

@app.route('/health')
def health_check():
    # Return 200 immediately so Render knows the server is UP
    # Even if models aren't loaded yet, the server is "alive"
    status = "ready" if model is not None else "loading_models"
    return jsonify({"status": "online", "model_status": status}), 200

@app.route('/ping')
def ping():
    # Trigger model loading in the background
    try:
        load_models_lazy()
        return jsonify({"status": "ready"}), 200
    except:
        return jsonify({"status": "error"}), 500

IMG_SIZE = 224

TUMOR_INFO = {
    'glioma': {
        'full_name': 'Glioma Tumor',
        'severity': 'High',
        'color': '#ef4444',
        'description': 'Gliomas are tumors that arise from glial cells in the brain or spine.',
        'treatment': 'Surgery, Radiation, Chemotherapy',
        'urgency': '⚠️ Immediate medical attention required'
    },
    'meningioma': {
        'full_name': 'Meningioma Tumor',
        'severity': 'Medium',
        'color': '#f97316',
        'description': 'Meningiomas arise from the meninges, the membranes surrounding the brain.',
        'treatment': 'Surgery, Radiation therapy',
        'urgency': '🔶 Medical consultation recommended'
    },
    'notumor': {
        'full_name': 'No Tumor Detected',
        'severity': 'None',
        'color': '#22c55e',
        'description': 'No tumor detected in the MRI scan.',
        'treatment': 'No treatment required',
        'urgency': '✅ Brain appears healthy'
    },
    'pituitary': {
        'full_name': 'Pituitary Tumor',
        'severity': 'Medium',
        'color': '#3b82f6',
        'description': 'Pituitary tumors form in the pituitary gland at the base of the brain.',
        'treatment': 'Surgery, Medication, Radiation',
        'urgency': '🔶 Medical consultation recommended'
    }
}

def preprocess_image(image_bytes):
    """Preprocess image for EfficientNet."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def generate_gradcam(img_array):
    """Generate Grad-CAM heatmap."""
    try:
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            pred_idx    = int(tf.argmax(preds[0]).numpy())
            class_score = preds[:, pred_idx]

        grads   = tape.gradient(class_score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize and colorize
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Original image for overlay
        orig = img_array[0].copy()
        orig = (orig - orig.min()) / (orig.max() - orig.min()) * 255
        orig = orig.astype(np.uint8)
        orig_bgr = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Convert to base64
        _, buffer = cv2.imencode('.png', overlay_rgb)
        gradcam_b64 = base64.b64encode(buffer).decode('utf-8')
        return gradcam_b64
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


@app.route('/', methods=['GET'])
def index():
    return """
    <h1>🧠 NeuraScan Backend API is Online</h1>
    <p>This is the API server for the Brain Tumor Classification project.</p>
    <p>To use the application, please visit the <a href="https://neurascan-frontend.onrender.com">Frontend Website</a>.</p>
    <p>Status: <b>Active</b></p>
    """


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Brain Tumor API is running!'})


@app.route('/predict', methods=['POST'])
def predict():
    gc.collect() # Pre-cleanup
    # Ensure models are loaded before processing
    load_models_lazy()

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file  = request.files['image']
    image_bytes = file.read()

    # Preprocess
    img_array = preprocess_image(image_bytes)

    # ── Single-pass DL Prediction & Feature Extraction ─────────────
    features_raw, dl_preds = prediction_model.predict(img_array, verbose=0)
    dl_preds = dl_preds[0]
    
    dl_idx     = int(np.argmax(dl_preds))
    dl_class   = CLASS_NAMES[dl_idx]
    dl_conf    = float(dl_preds[dl_idx])
    
    # ── Hybrid SVM Prediction ──────────────────────────────────────
    features   = scaler.transform(features_raw)
    svm_idx    = int(svm.predict(features)[0])
    svm_probs  = svm.predict_proba(features)[0]
    svm_class  = CLASS_NAMES[svm_idx]
    svm_conf   = float(svm_probs[svm_idx])

    # Final prediction (use whichever is more confident)
    final_class = dl_class if dl_conf >= svm_conf else svm_class
    final_conf  = max(dl_conf, svm_conf)

    # All class probabilities
    all_probs = {CLASS_NAMES[i]: round(float(dl_preds[i]) * 100, 2)
                 for i in range(len(CLASS_NAMES))}

    # Grad-CAM (Heatmap) - DISABLED in Slim Mode for stability
    gradcam_img = None
    # try:
    #     gradcam_img = generate_gradcam(img_array)
    # except Exception as e:
    #     print(f"Skipping Grad-CAM due to memory: {e}")
    #     gradcam_img = None

    # Tumor info
    info = TUMOR_INFO.get(final_class, {})

    # Clear memory explicitly to prevent Free Tier crashes
    del img_array
    del features_raw
    gc.collect()

    return jsonify({
        'prediction':   final_class,
        'confidence':   round(final_conf * 100, 2),
        'dl_prediction':  dl_class,
        'dl_confidence':  round(dl_conf * 100, 2),
        'svm_prediction': svm_class,
        'svm_confidence': round(svm_conf * 100, 2),
        'all_probabilities': all_probs,
        'gradcam': gradcam_img,
        'tumor_info': info
    })


if __name__ == '__main__':
    print("🚀 Starting Brain Tumor API on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

import tensorflow as tf
import os

BASE_DIR = r"c:\Users\vishn\Downloads\Brain tumor\brain-tumor-app\backend"
model_path = os.path.join(BASE_DIR, 'brain_tumor_97.keras')

try:
    model = tf.keras.models.load_model(model_path)
    print("Layer Names:")
    for layer in model.layers:
        print(layer.name)
except Exception as e:
    print(f"Error: {e}")

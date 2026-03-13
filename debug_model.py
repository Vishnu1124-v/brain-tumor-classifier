import tensorflow as tf
import os

BASE_DIR = r"c:\Users\vishn\Downloads\Brain tumor\brain-tumor-app\backend"
model_path = os.path.join(BASE_DIR, 'brain_tumor_97.keras')

try:
    model = tf.keras.models.load_model(model_path)
    print("Model Loaded Successfully")
    print("\nLast 10 layers:")
    for layer in model.layers[-10:]:
        print(f"Layer: {layer.name}, Type: {type(layer)}")
    
    # Check if 'top_conv' exists
    try:
        layer = model.get_layer('top_conv')
        print(f"\n'top_conv' found! Output shape: {layer.output_shape}")
    except ValueError:
        print("\n'top_conv' NOT found.")
        
except Exception as e:
    print(f"Error: {e}")

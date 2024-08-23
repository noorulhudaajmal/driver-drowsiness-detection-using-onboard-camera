import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Directory paths
models_dir = 'models'
tflite_models_dir = 'models-tflite'

# check if the output directory exists
os.makedirs(tflite_models_dir, exist_ok=True)

def convert_to_tflite(model_path, tflite_model_path):
    """
    Convert a Keras model to TensorFlow Lite format and save it.

    Args:
    - model_path (str): Path to the Keras model file.
    - tflite_model_path (str): Path to save the TensorFlow Lite model.
    """
    # Loading in the Keras model
    model = load_model(model_path)

    # model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the converted model to file
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

def main():
    # List all Keras model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]

    # Convert each model to TensorFlow Lite format
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        tflite_model_path = os.path.join(tflite_models_dir, model_file.replace('.keras', '.tflite'))
        convert_to_tflite(model_path, tflite_model_path)
        print(f'Converted {model_file} to TensorFlow Lite format and saved to {tflite_model_path}')

if __name__ == '__main__':
    main()

import tensorflow as tf
from keras.models import load_model
import cv2 as cv
import numpy as np


# m1 = load_model("models/blink_model.h5")
# m2 = load_model("models/yawn_model.h5")
#
# # Convert the model
# converter = tf.lite.TFLiteConverter.from_keras_model(m1) # path to the SavedModel directory
# tflite_model = converter.convert()
#
# # Save the model.
# with open('models/blinking_model.tflite', 'wb') as f:
#   f.write(tflite_model)
#
# # Convert the model
# converter = tf.lite.TFLiteConverter.from_keras_model(m2) # path to the SavedModel directory
# tflite_model = converter.convert()
#
# # Save the model.
# with open('models/yawning_model.tflite', 'wb') as f:
#   f.write(tflite_model)


m1 = load_model("Blink_model0.h5")

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(m1) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('Blink_model0.tflite', 'wb') as f:
  f.write(tflite_model)

# # Convert the model
# converter = tf.lite.TFLiteConverter.from_keras_model(m2) # path to the SavedModel directory
# tflite_model = converter.convert()
#
# # Save the model.
# with open('models/yawning_model.tflite', 'wb') as f:
#   f.write(tflite_model)

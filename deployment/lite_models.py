import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.models import load_model
from utils import preprocess_data, get_gray, get_facial_landmarks


class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data):
        input_data = input_data.astype(np.float32)
        input_data = preprocess_data(input_data)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][0]


class YawningModel(TFLiteModel):
    def __init__(self, model_path):
        super().__init__(model_path)


class BlinkingModel(TFLiteModel):
    def __init__(self, model_path):
        super().__init__(model_path)

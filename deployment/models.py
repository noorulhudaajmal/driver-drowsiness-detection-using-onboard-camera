import cv2 as cv
import numpy as np
from utils import *
from keras.models import load_model


class YawningModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, input_data):
        # input_data = cv.resize(input_data, (256, 256))
        input_data = input_data/255.0
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        return self.model.predict(input_data)[0][0]

class BlinkingModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, input_data):
        # input_data = cv.resize(input_data, (256, 256))
        input_data = input_data/255.0
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        return self.model.predict(input_data)[0][0]


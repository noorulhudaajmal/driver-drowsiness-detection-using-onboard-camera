import numpy as np
import tensorflow as tf
import cv2 as cv

import tensorflow as tf
import cv2 as cv
import numpy as np
import dlib


def get_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("detectors/shape_predictor_68_face_landmarks.dat")

    def detect(self, image):
        # gray = get_gray(image)
        face_rectangles = self.detector(image, 1)
        if len(face_rectangles) == 0:
            return -1
        face_landmarks = self.predictor(image, face_rectangles[0])
        face_landmarks = np.array([[p.x, p.y] for p in face_landmarks.parts()])
        return face_landmarks


class YawningModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data):
        input_shape = self.input_details[0]['shape']
        input_data = input_data.astype(np.float32)
        input_data = cv.cvtColor(input_data, cv.COLOR_BGR2GRAY)
        input_data = cv.resize(input_data, (256, 256))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][0]

    def get_roi(self, landmarks, roi_keypoints):
        x_min = np.min(landmarks[roi_keypoints][:, 0])
        x_max = np.max(landmarks[roi_keypoints][:, 0])
        y_min = np.min(landmarks[roi_keypoints][:, 1])
        y_max = np.max(landmarks[roi_keypoints][:, 1])
        return x_min, x_max, y_min, y_max

    def get_mouth_roi(self, landmarks):
        mouth_roi = self.get_roi(landmarks, list(range(48, 68)))
        return mouth_roi

    def run(self, image):
        print(image.shape)
        return self.predict(image)
        # image = get_gray(image)
        #
        # detector = FaceDetector()
        # s = detector.detect(image)
        #
        # if isinstance(s, np.ndarray):
        #     mouth_roi = self.get_mouth_roi(s)
        #     mouth_yawn = self.predict(image[mouth_roi[2]:mouth_roi[3], mouth_roi[0]:mouth_roi[1]])
        #
        #     return mouth_yawn
        # return None


model = YawningModel("models/yawning_model.tflite")
result = model.run(cv.imread("mouth_115.png"))
print(result)


#
# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="models/yawning_model.tflite")
# interpreter.allocate_tensors()
#
# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Test model on random input data.
# input_shape = input_details[0]['shape']
#
# input_data = cv.imread("tetss/4.jpg")
# input_data = input_data.astype(np.float32)
#
# print(input_data)
# input_data = cv.cvtColor(input_data, cv.COLOR_BGR2GRAY)
#
# input_data = cv.resize(input_data, (256, 256))
# print(input_data.shape)
#
# input_data = np.expand_dims(input_data, axis=0)
# input_data = np.expand_dims(input_data, axis=-1)
# print(input_data.shape)
# # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)
#
# interpreter.invoke()
#
# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data[0][0])




import dlib
import numpy as np
from utils import get_gray, get_facial_landmarks, crop_region, resize_image


class FaceDetector:
    def __init__(self, shape_predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """Detects face landmarks in an image."""
        gray_image = get_gray(image)

        return get_facial_landmarks(gray_image)

    def get_eyes_roi(self, image, landmarks):
        """Extracts the left and right eye regions based on landmarks."""
        l_eye_roi = crop_region(image, (landmarks[18][0] - 8, landmarks[18][1] - 8),
                                (landmarks[21][0] + 5, landmarks[41][1] + 8))
        r_eye_roi = crop_region(image, (landmarks[22][0] - 8, landmarks[22][1] - 9),
                                (landmarks[25][0] + 5, landmarks[46][1] + 5))

        return resize_image(l_eye_roi), resize_image(r_eye_roi)

    def get_mouth_roi(self, image, landmarks):
        """Extracts the mouth region based on landmarks."""
        mouth_roi = crop_region(image, (landmarks[48][0] - 5, landmarks[51][1] - 7),
                                (landmarks[54][0] + 5, landmarks[57][1] + 8))

        return resize_image(mouth_roi)

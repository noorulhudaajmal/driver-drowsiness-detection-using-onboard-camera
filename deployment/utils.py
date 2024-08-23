import cv2 as cv
import dlib
import numpy as np
import os


# Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../training/assets/detector/shape_predictor_68_face_landmarks.dat")


def get_gray(img):
    """Convert an image to grayscale."""
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def get_facial_landmarks(gray):
    """Extract facial landmarks from a grayscale image."""
    face_rectangles = detector(gray, 1)
    if len(face_rectangles) == 0:
        return -1
    face_landmarks = predictor(gray, face_rectangles[0])
    return np.array([[p.x, p.y] for p in face_landmarks.parts()])


def beep():
    """Trigger a system beep."""
    os.system("beep -f 1000q -l 1500")


def preprocess_data(input_data):
    """Preprocess the input data for model prediction."""
    input_data = input_data / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    return input_data


def crop_region(image, start_point, end_point):
    """Crop a region from an image."""
    return image[start_point[1]:end_point[1], start_point[0]:end_point[0]]


def resize_image(image, size=(256, 256)):
    """Resize an image to the given size."""
    return cv.resize(image, size) if image.size != 0 else np.array([])


def label_face(image, landmarks):
    """Draw rectangles around the mouth, left eye, and right eye regions."""
    regions = {
        'mouth': [(landmarks[48][0] - 5, landmarks[51][1] - 5), (landmarks[54][0] + 5, landmarks[57][1] + 8)],
        'left_eye': [(landmarks[18][0] - 8, landmarks[18][1] - 8), (landmarks[21][0] + 5, landmarks[41][1] + 8)],
        'right_eye': [(landmarks[22][0] - 8, landmarks[22][1] - 9), (landmarks[25][0] + 5, landmarks[46][1] + 5)],
    }

    for region, (start, end) in regions.items():
        image = cv.rectangle(image, start, end, color=(255, 0, 0), thickness=2)

    return image

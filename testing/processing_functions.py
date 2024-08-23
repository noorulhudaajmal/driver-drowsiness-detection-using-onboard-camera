import numpy as np
import cv2 as cv
import os


# Function to convert an image to grayscale
def get_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Function to extract facial landmarks
def get_facial_landmarks(gray, detector, predictor):
    face_rectangles = detector(gray, 1)
    if len(face_rectangles) == 0:
        return -1
    face_landmarks = predictor(gray, face_rectangles[0])
    face_landmarks = np.array([[p.x, p.y] for p in face_landmarks.parts()])
    return face_landmarks


# Function to get a frame from a video at a specific time (in seconds)
def get_frame(second, capture):
    capture.set(cv.CAP_PROP_POS_MSEC, second * 1000)
    has_frames, img = capture.read()
    return has_frames, img


# Function to trigger a system beep
def beep():
    os.system("beep -f 1000q -l 1500")


# Function to detect face landmarks in an image and return cropped region
def process_image(image, detector, predictor):
    gray_image = get_gray(image)
    landmarks = get_facial_landmarks(gray_image, detector, predictor)

    if isinstance(landmarks, int) and landmarks == -1:
        print("No face detected")
        return landmarks, np.array([]), np.array([]), np.array([])

    # Crop the mouth region
    mx = landmarks[48][0] - 5
    mxw = landmarks[54][0] + 5
    my = landmarks[51][1] - 7
    myh = landmarks[57][1] + 8
    mouth = gray_image[my:myh, mx:mxw]

    # Crop the left eye region
    lx = landmarks[18][0] - 8
    lxw = landmarks[21][0] + 5
    ly = landmarks[18][1] - 8
    lyh = landmarks[41][1] + 8
    left_eye = gray_image[ly:lyh, lx:lxw]

    # Crop the right eye region
    rx = landmarks[22][0] - 8
    rxw = landmarks[25][0] + 5
    ry = landmarks[22][1] - 9
    ryh = landmarks[46][1] + 5
    right_eye = gray_image[ry:ryh, rx:rxw]

    # resizing
    mouth = cv.resize(mouth, (256, 256)) if mouth.size != 0 else np.array([])
    left_eye = cv.resize(left_eye, (256, 256)) if left_eye.size != 0 else np.array([])
    right_eye = cv.resize(right_eye, (256, 256)) if right_eye.size != 0 else np.array([])

    return landmarks, mouth, left_eye, right_eye


# Draw rectangles around the mouth, left eye, and right eye regions.
def get_labeled_face(image, landmarks):
    # Mouth
    mouth = cv.rectangle(image,
                          (landmarks[48][0] - 5, landmarks[51][1] - 5),
                          (landmarks[54][0] + 5, landmarks[57][1] + 8),
                          color=(255, 0, 0), thickness=2)
    # Left Eye
    left_eye = cv.rectangle(image,
                             (landmarks[18][0] - 8, landmarks[18][1] - 8),
                             (landmarks[21][0] + 5, landmarks[41][1] + 8),
                             color=(0, 255, 0), thickness=2)

    # Right Eye
    right_eye = cv.rectangle(image,
                              (landmarks[22][0] - 8, landmarks[22][1] - 9),
                              (landmarks[25][0] + 5, landmarks[46][1] + 5),
                              color=(0, 0, 255), thickness=2)

    return image, mouth, left_eye, right_eye


def preprocess_data(input_data):
    input_data = input_data/255.0
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)

    return input_data
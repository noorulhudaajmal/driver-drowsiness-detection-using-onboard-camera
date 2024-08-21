import cv2 as cv
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/huda/Desktop/fyp files/FYP/detectors/shape_predictor_68_face_landmarks.dat")


def get_facial_landmarks(gray):
    face_rectangles = detector(gray, 1)
    if len(face_rectangles) == 0:
        return -1
    face_landmarks = predictor(gray, face_rectangles[0])
    face_landmarks = np.array([[p.x, p.y] for p in face_landmarks.parts()])
    return face_landmarks


def get_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


print("When there is a face in the input image")
# read an image with a face
img = cv.imread("img.png")
# conversion to gray scale
img = get_gray(img)
# detect facial landmarks using the function
landmarks = get_facial_landmarks(img)
# verify that the output is an array of landmarks
assert isinstance(landmarks, np.ndarray)
assert landmarks.ndim == 2
assert landmarks.shape[1] == 2

# verify that the landmarks are within the expected range
assert landmarks[:, 0].min() >= 0 and landmarks[:, 0].max() < img.shape[1]
assert landmarks[:, 1].min() >= 0 and landmarks[:, 1].max() < img.shape[0]

print("When no face is detected")

# create an image with no faces
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv.imshow("no face",img)
cv.waitKey()
# detect facial landmarks using the function
landmarks = get_facial_landmarks(img)

# verify that the output is -1 (indicating no face was detected)
assert landmarks == -1

import numpy as np
import cv2 as cv
import dlib
import os
from processing_functions import get_gray, get_facial_landmarks, get_frame, beep, preprocess_data, get_labeled_face, \
    process_image


# Load models and initialize variables
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../training/assets/detector/shape_predictor_68_face_landmarks.dat")
cap = cv.VideoCapture(r'test-sets/video_input/1.mp4')


# Test 1: Grayscale conversion
def test_get_gray():
    # Test with red image
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :, 2] = 255  # Red channel full intensity
    gray_img = get_gray(color_img)
    assert gray_img.ndim == 2
    assert np.array_equal(gray_img, np.zeros((100, 100), dtype=np.uint8) + 76)  # 0.299 * 255 = 76

    # Test with green image
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :, 1] = 255  # Green channel full intensity
    gray_img = get_gray(color_img)
    assert gray_img.ndim == 2
    assert np.array_equal(gray_img, np.zeros((100, 100), dtype=np.uint8) + 150)  # 0.587 * 255 = 150


# Test 2: Facial landmarks extraction
def test_get_facial_landmarks():
    # Test with an image containing a face
    print("When there is a face in the input image")
    img = cv.imread("test-sets/image.jpeg")
    img_gray = get_gray(img)
    landmarks = get_facial_landmarks(img_gray, detector, predictor)
    assert isinstance(landmarks, np.ndarray)
    assert landmarks.ndim == 2
    assert landmarks.shape[1] == 2
    assert landmarks[:, 0].min() >= 0 and landmarks[:, 0].max() < img_gray.shape[1]
    assert landmarks[:, 1].min() >= 0 and landmarks[:, 1].max() < img_gray.shape[0]

    # Test with an image with no faces
    print("When no face is detected")
    img_no_face = np.zeros((100, 100, 3), dtype=np.uint8)
    cv.imshow("No Face Image", img_no_face)
    cv.waitKey(0)
    landmarks = get_facial_landmarks(get_gray(img_no_face),  detector, predictor)
    assert landmarks == -1


# Test 3: Video frame extraction
def test_get_frame():
    cap.set(3, 440)  # width
    cap.set(4, 280)  # height
    cap.set(10, 50)  # brightness

    success, frame = get_frame(1, capture=cap)
    print(type(cap))
    print(success)
    assert success
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3


# Test 4: System beep call
def test_beep():
    call_log = []
    def mock_system_call(cmd):
        call_log.append(cmd)
    os.system = mock_system_call
    beep()
    assert len(call_log) == 1
    assert call_log[0] == "beep -f 1000q -l 1500"


# Test 5:
def test_process_image():
    # Test with an image containing a face
    img = cv.imread("test-sets/image.jpeg")
    landmarks, mouth, left_eye, right_eye = process_image(img, detector, predictor)

    assert isinstance(landmarks, np.ndarray)
    assert mouth.shape == (256, 256) and left_eye.shape == (256, 256) and right_eye.shape == (256, 256)

    # Test with an image with no faces
    img_no_face = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks, mouth, left_eye, right_eye = process_image(img_no_face, detector, predictor)

    assert landmarks == -1
    assert mouth.size == 0 and left_eye.size == 0 and right_eye.size == 0


# Test: 6
def test_get_labeled_face():
    img = cv.imread("test-sets/image.jpeg")
    landmarks = get_facial_landmarks(get_gray(img), detector, predictor)

    labeled_image, mouth_rect, left_eye_rect, right_eye_rect = get_labeled_face(img, landmarks)

    assert isinstance(labeled_image, np.ndarray)
    assert labeled_image.shape == img.shape

    # Check if the rectangles were drawn
    assert (mouth_rect.shape == img.shape)
    assert (left_eye_rect.shape == img.shape)
    assert (right_eye_rect.shape == img.shape)


# Test: 7
def test_preprocess_data():
    # Create a dummy 256x256 grayscale image
    input_data = np.ones((256, 256), dtype=np.uint8) * 255

    preprocessed_data = preprocess_data(input_data)

    assert preprocessed_data.shape == (1, 256, 256, 1)
    assert np.all(preprocessed_data <= 1.0)  # normalize data between 0 and 1
    assert np.all(preprocessed_data >= 0.0)



# running all tests
if __name__ == '__main__':
    test_get_gray()
    test_get_facial_landmarks()
    test_get_frame()
    test_beep()
    test_process_image()
    test_get_labeled_face()
    test_preprocess_data()
    print("All tests passed!")


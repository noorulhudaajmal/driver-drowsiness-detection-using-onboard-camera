import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# Load the Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/detector/shape_predictor_68_face_landmarks.dat")


def get_gray(img):
    """
    Convert the image to grayscale.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_facial_landmarks(gray):
    """
    Detect face landmarks in the given grayscale image.
    """
    face_rects = detector(gray, 1)
    if len(face_rects) == 0:
        return -1
    face_landmarks = predictor(gray, face_rects[0])
    face_landmarks = np.array([[p.x, p.y] for p in face_landmarks.parts()])
    return face_landmarks


def eye_aspect_ratio(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR).
    """
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth_landmarks):
    """
    Calculate the Mouth Aspect Ratio (MAR).
    """
    A = np.linalg.norm(mouth_landmarks[13] - mouth_landmarks[19])
    B = np.linalg.norm(mouth_landmarks[14] - mouth_landmarks[18])
    C = np.linalg.norm(mouth_landmarks[15] - mouth_landmarks[17])
    D = np.linalg.norm(mouth_landmarks[12] - mouth_landmarks[16])
    mar = (A + B + C) / (2.0 * D)
    return mar


def facial_asymmetry(facial_landmarks):
    """
    Calculate the facial asymmetry as the standard deviation of the distances
    from the face center to the landmarks.
    """
    face_center = np.mean(facial_landmarks, axis=0)
    distances = np.linalg.norm(facial_landmarks - face_center, axis=1)
    asymmetry = np.std(distances)
    return asymmetry


def get_features(landmarks):
    """
    Extract the Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
    and Facial Asymmetry (FA) from the facial landmarks.
    """
    eye_landmarks = landmarks[36:42]
    mouth_landmarks = landmarks[48:68]
    ear = eye_aspect_ratio(eye_landmarks)
    mar = mouth_aspect_ratio(mouth_landmarks)
    fa = facial_asymmetry(landmarks)
    return ear, mar, fa


def plot_landmarks(image, landmarks, color='blue'):
    """
    Plot the landmarks on the image.
    """
    plt.imshow(image)
    for i in range(landmarks.shape[0]):
        plt.scatter(landmarks[i][0], landmarks[i][1], color=color)
    plt.show()


def get_labeled_face(image, landmarks):
    """
    Draw rectangles around the mouth, left eye, and right eye regions.
    """
    # Mouth
    mouth = cv2.rectangle(image,
                          (landmarks[48][0] - 5, landmarks[51][1] - 5),
                          (landmarks[54][0] + 5, landmarks[57][1] + 8),
                          color=(255, 0, 0), thickness=2)

    # Left Eye
    left_eye = cv2.rectangle(image,
                             (landmarks[36][0] - 5, landmarks[36][1] - 5),
                             (landmarks[39][0] + 5, landmarks[41][1] + 8),
                             color=(0, 255, 0), thickness=2)

    # Right Eye
    right_eye = cv2.rectangle(image,
                              (landmarks[42][0] - 5, landmarks[42][1] - 5),
                              (landmarks[45][0] + 5, landmarks[46][1] + 5),
                              color=(0, 0, 255), thickness=2)

    return image, mouth, left_eye, right_eye


def plot_labeled_face_image(image_path):
    """
    Process the image to detect landmarks,
    and display the labeled image.
    """
    # Read the image from the specified path
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = get_gray(img)

    # Detect facial landmarks
    landmarks = get_facial_landmarks(gray_image)

    if isinstance(landmarks, int) and landmarks == -1:
        print("No face detected")
        plt.imshow(img)
        plt.text(0.5, 0.5, 'No Face Detected', fontsize=18, ha='center')
        plt.axis('off')
        plt.show()
        return

    # Draw rectangles around the mouth, left eye, and right eye
    labeled_image, _, _, _ = get_labeled_face(img, landmarks)

    # Display the labeled image
    plt.imshow(cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def process_image(image):
    """
    Process the image to detect landmarks and return cropped regions of interest.

    Args:
        image (numpy.ndarray): image array.

    Returns:
        mouth (numpy.ndarray): The cropped mouth region.
        left_eye (numpy.ndarray): The cropped left eye region.
        right_eye (numpy.ndarray): The cropped right eye region.
    """
    # image = cv2.imread(image_path)
    gray_image = get_gray(image)
    landmarks = get_facial_landmarks(gray_image)

    if isinstance(landmarks, int) and landmarks == -1:
        print("No face detected")
        return np.array([]), np.array([]), np.array([])

    # Crop the mouth region
    mx = landmarks[48][0] - 5
    mxw = landmarks[54][0] + 5
    my = landmarks[51][1] - 7
    myh = landmarks[57][1] + 8
    mouth = image[my:myh, mx:mxw]

    # Crop the left eye region
    lx = landmarks[18][0] - 8
    lxw = landmarks[21][0] + 5
    ly = landmarks[18][1] - 8
    lyh = landmarks[41][1] + 8
    left_eye = image[ly:lyh, lx:lxw]

    # Crop the right eye region
    rx = landmarks[22][0] - 8
    rxw = landmarks[25][0] + 5
    ry = landmarks[22][1] - 9
    ryh = landmarks[46][1] + 5
    right_eye = image[ry:ryh, rx:rxw]

    # resizing
    mouth = cv2.resize(mouth, (256, 256)) if mouth.size != 0 else np.array([])
    left_eye = cv2.resize(left_eye, (256, 256)) if left_eye.size != 0 else np.array([])
    right_eye = cv2.resize(right_eye, (256, 256)) if right_eye.size != 0 else np.array([])

    return mouth, left_eye, right_eye


def plot_image_with_fallback(image, title):
    """
    Plot the image if it exists, otherwise plot an empty figure with a message.
    """
    if image.size == 0:
        plt.figure()
        plt.text(0.5, 0.5, 'Region not detected', fontsize=18, ha='center')
        plt.title(title)
        plt.axis('off')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":

    TEST_IMG_PATH = 'facial-landmark-extraction/test-images/angry.jpeg'

    plot_labeled_face_image(TEST_IMG_PATH)

    mouth, left_eye, right_eye = process_image(cv2.imread(TEST_IMG_PATH))

    plot_image_with_fallback(mouth, "Mouth")
    plot_image_with_fallback(left_eye, "Left Eye")
    plot_image_with_fallback(right_eye, "Right Eye")


# mx = s[48][0]-5
# mxw = s[54][0]+5
# my = s[51][1]-7
# myh = s[57][1]+8

# lx = s[18][0]-8
# lxw = s[21][0]+5
# ly = s[18][1]-8
# lyh = s[41][1]+8

# rx = s[22][0]-8
# rxw = s[25][0]+5
# ry = s[22][1]-9
# ryh = s[46][1]+5
# mouth = image[ my:myh , mx:mxw ]
# l_eye = image[ ly:lyh , lx:lxw ]
# r_eye = image[ ry:ryh , rx:rxw ]

# l_eye = cv2.resize(l_eye, (256,256))
# r_eye = cv2.resize(r_eye, (256,256))
# mouth = cv2.resize(mouth, (256,256))
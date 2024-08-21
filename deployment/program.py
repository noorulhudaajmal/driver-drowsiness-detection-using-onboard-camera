import cv2 as cv
import dlib
import numpy as np
import os
from keras.models import load_model
import threading


def beep():
    os.system("beep -f 1000 -l 3000")


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


class YawnDetector:
    def __init__(self):
        self.model = load_model("models/yawn_model.h5")

    def predict(self, image):
        image = cv.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)[0][0]
        return prediction


class BlinkDetector:
    def __init__(self):
        self.model = load_model("models/blink_model.h5")

    def predict(self, image):
        image = cv.resize(image, (256, 256))
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)[0][0]
        return prediction


class DriverMonitor:
    def __init__(self):
        self.blink_model = BlinkDetector()
        self.yawn_model = YawnDetector()
        self.face_detector = FaceDetector()

    def get_roi(self, landmarks, roi_keypoints):
        x_min = np.min(landmarks[roi_keypoints][:, 0])
        x_max = np.max(landmarks[roi_keypoints][:, 0])
        y_min = np.min(landmarks[roi_keypoints][:, 1])
        y_max = np.max(landmarks[roi_keypoints][:, 1])
        return x_min, x_max, y_min, y_max

    def get_eyes_roi(self, landmarks):
        l_eye_roi = self.get_roi(landmarks, list(range(36, 42)))
        r_eye_roi = self.get_roi(landmarks, list(range(42, 48)))
        return l_eye_roi, r_eye_roi

    def get_mouth_roi(self, landmarks):
        mouth_roi = self.get_roi(landmarks, list(range(48, 68)))
        return mouth_roi

    def predict_blink(self, eye_roi):
        return self.blink_model.predict(eye_roi)

    def predict_yawn(self, mouth_roi):
        return self.yawn_model.predict(mouth_roi)

    def run(self, video_path):
        cap = cv.VideoCapture(video_path)
        cap.set(3, 440)  # width
        cap.set(4, 280)  # height
        cap.set(10, 50)  # brightness

        blinks = []
        yawns = []

        sec = 0
        frame_rate = 1
        success, image = cap.read()
        per_close = 0

        # start beeping thread
        beep_thread = threading.Thread(target=beep)
        # beep_thread.start()
        o=1
        while success:
            sec = sec + frame_rate
            sec = round(sec, 2)
            image = get_gray(image)
            s = self.face_detector.detect(image)

            if isinstance(s, np.ndarray):
                l_eye_roi, r_eye_roi = self.get_eyes_roi(s)
                mouth_roi = self.get_mouth_roi(s)

                l_eye_blink = self.predict_blink(image[l_eye_roi[2]:l_eye_roi[3], l_eye_roi[0]:l_eye_roi[1]])
                r_eye_blink = self.predict_blink(image[r_eye_roi[2]:r_eye_roi[3], r_eye_roi[0]:r_eye_roi[1]])
                mouth_yawn = self.predict_yawn(image[mouth_roi[2]:mouth_roi[3], mouth_roi[0]:mouth_roi[1]])

                cv.imwrite(f"mouth_{o}.png", image[mouth_roi[2]:mouth_roi[3], mouth_roi[0]:mouth_roi[1]])
                o+=1

                # blinks.append(l_eye_blink and r_eye_blink)
                blinks.append(r_eye_blink)
                blinks.append(l_eye_blink)

                yawns.append(mouth_yawn)

                mouth = cv.rectangle(image, (s[48][0] - 5, s[51][1] - 5), (s[54][0] + 5, s[57][1] + 8), color=(255, 1, 1))
                # left eye
                left_eye = cv.rectangle(image, (s[18][0] - 5, s[18][1] - 5), (s[21][0] + 5, s[41][1] + 8), color=(255, 1, 1))
                # right eye
                right_eye = cv.rectangle(image, (s[22][0] - 5, s[22][1] - 5), (s[25][0] + 5, s[46][1] + 5), color=(255, 1, 1))

                if len(blinks) >= 5:
                    b = blinks[-5:]
                    b = np.ceil(b)
                    w = yawns[-5:]
                    w = np.ceil(w)
                    per_close = 0.75 * (list(b).count(1) / len(b)) + (0.25 * (list(w).count(1) / len(w)))

                if per_close > 0.55:
                    # start beeping thread if not already started
                    # if not beep_thread.is_alive():
                    beep_thread = threading.Thread(target=beep)
                    beep_thread.start()
                    image = cv.putText(image, " --- DROWSY", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                   color=(255, 1, 1))

                cv.imshow("Results", image)

            success, image = cap.read()

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv.destroyAllWindows()


monitor = DriverMonitor()
#
monitor.run(r'test-sets/ghi.mp4')

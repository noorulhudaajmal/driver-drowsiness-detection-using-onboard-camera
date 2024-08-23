import cv2 as cv
import numpy as np
from collections import deque
from threading import Thread

from face_detector import FaceDetector
from lite_models import YawningModel, BlinkingModel
from testing.processing_functions import get_gray
from utils import beep, label_face


class DriverMonitor:
    def __init__(self, blink_model_path, yawn_model_path, shape_predictor_path):
        self.face_detector = FaceDetector(shape_predictor_path)
        self.blink_model = BlinkingModel(blink_model_path)
        self.yawn_model = YawningModel(yawn_model_path)

        # Queue for storing blink and yawn states
        queue_size = 20
        self.blinks_queue = deque([0] * queue_size, maxlen=queue_size)
        self.yawns_queue = deque([0] * queue_size, maxlen=queue_size)

        # Weights for drowsiness detection
        self.blink_weight = 0.70
        self.yawn_weight = 0.30
        self.DROWSINESS_THRESHOLD = 0.65

        # Warm-up period
        self.warmup_frames = 5
        self.frame_count = 0

    def predict_blink(self, eye_roi):
        return self.blink_model.predict(get_gray(eye_roi))

    def predict_yawn(self, mouth_roi):
        return self.yawn_model.predict(get_gray(mouth_roi))

    def detect_drowsiness(self, image, landmarks):
        l_eye_roi, r_eye_roi = self.face_detector.get_eyes_roi(image, landmarks)
        mouth_roi = self.face_detector.get_mouth_roi(image, landmarks)

        l_eye_blink = int(self.predict_blink(l_eye_roi) < 0.5)
        r_eye_blink = int(self.predict_blink(r_eye_roi) < 0.5)
        is_yawning = int(self.predict_yawn(mouth_roi) > 0.5)

        # Updating queues
        self.blinks_queue.append(l_eye_blink)
        self.blinks_queue.append(r_eye_blink)
        self.yawns_queue.append(is_yawning)

        # the weighted rolling average
        blink_avg = sum(self.blinks_queue) / len(self.blinks_queue)
        yawn_avg = sum(self.yawns_queue) / len(self.yawns_queue)

        # drowsiness score
        drowsiness_score = (self.blink_weight * blink_avg) + (self.yawn_weight * yawn_avg)

        return drowsiness_score

    def run(self, video_path):
        cap = cv.VideoCapture(video_path)
        cap.set(3, 440)  # width
        cap.set(4, 280)  # height
        cap.set(10, 50)  # brightness

        while True:
            success, image = cap.read()
            if not success:
                break

            landmarks = self.face_detector.detect(image)
            if isinstance(landmarks, np.ndarray):
                drowsiness_score = self.detect_drowsiness(image, landmarks)
                labeled_image = label_face(image, landmarks)

                if self.frame_count > self.warmup_frames:
                    if drowsiness_score > self.DROWSINESS_THRESHOLD:
                        Thread(target=beep).start()
                        labeled_image = cv.putText(labeled_image, " --- DROWSY", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                                   fontScale=1, color=(255, 1, 1))
                    else:
                        labeled_image = cv.putText(labeled_image, " --- ALERT", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                                   fontScale=1, color=(0, 255, 0))
                else:
                    labeled_image = cv.putText(labeled_image, " --- INITIALIZING", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                               fontScale=1, color=(255, 255, 0))
            else:
                labeled_image = cv.putText(image, "DRIVER IS NOT FACING FRONT", (150, 150), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                           fontScale=1, color=(0, 0, 255))

            cv.imshow("Driver Monitoring", labeled_image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # model and shape predictor paths
    blink_model_path = "models/blink_model.tflite"
    yawn_model_path = "models/yawn_model.tflite"
    shape_predictor_path = "../training/assets/detector/shape_predictor_68_face_landmarks.dat"

    monitor = DriverMonitor(blink_model_path, yawn_model_path, shape_predictor_path)
    monitor.run("video_input/1.mp4")
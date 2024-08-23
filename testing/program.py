# import dlib
# import cv2 as cv
# import threading
# import numpy as np
# from collections import deque
# from processing_functions import process_image, preprocess_data, get_labeled_face, beep
# from keras.models import load_model
#
#
# # Dlib face predictors
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("../training/assets/detector/shape_predictor_68_face_landmarks.dat")
#
# # trained models
# yawning_model = load_model("../training/models/yawn_model.keras")
# blinking_model = load_model("../training/models/blink_model.keras")
#
# # Testing on video file input
# cap = cv.VideoCapture(r'test-sets/video_input/1.mp4')
# cap.set(3, 440)  # width
# cap.set(4, 280)  # height
# cap.set(10, 50)  # brightness
#
# sec = 0
# frameRate = 0.5
# fps = 300
# delay_time = int(1000 / fps)
# success, image = cap.read()
#
#
# # Queue for storing blink and yawn states
# queue_size = 20 # size of the queue
#
# # Initialize the deque (and fill with zeros initially)
# blinks_queue = deque([0] * queue_size, maxlen=queue_size)
# yawns_queue = deque([0] * queue_size, maxlen=queue_size)
#
# # Weights for blinking and yawning
# blink_weight = 0.70
# yawn_weight = 0.30
#
# # Threshold for drowsiness score
# DROWSINESS_THRESHOLD = 0.65
#
# # Minimum blink average to detect frequent blinking (microsleep)
# frequent_blink_threshold = 0.40
#
# # Warm-up period counter -> number of frames to skip before starting drowsiness detection
# warmup_frames = 5  # 5 seconds warm-up period
# frame_count = 0
#
# # Input Processing
# while success:
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     frame_count += 1
#
#     # extracting regions of interest from the frame
#     landmarks, mouth, l_eye, r_eye = process_image(image, detector, predictor)
#     if isinstance(landmarks, np.ndarray):
#         # preprocessing
#         mouth, l_eye, r_eye = preprocess_data(mouth), preprocess_data(l_eye), preprocess_data(r_eye)
#
#         # inferencing
#         l_blink = blinking_model.predict(l_eye)[0][0]
#         r_blink = blinking_model.predict(r_eye)[0][0]
#         ywn = yawning_model.predict(mouth)[0][0]
#
#         # log
#         print(f"Left Eye: {l_blink}")
#         print(f"Right Eye: {r_blink}")
#         print(f"Yawn: {ywn}")
#
#         # Blink detection {0: 'close', 1: 'open'}
#         l_eye_blink = (l_blink < 0.5).astype(int)
#         r_eye_blink = (r_blink < 0.5).astype(int)
#         # Yawn detection {0: 'no yawn', 1: 'yawn'}
#         is_yawning = (ywn > 0.5).astype(int)
#
#         # appending states to respective queues
#         blinks_queue.append(l_eye_blink)
#         blinks_queue.append(r_eye_blink)
#         yawns_queue.append(is_yawning)
#
#         # weighted rolling average
#         blink_avg = sum(blinks_queue) / len(blinks_queue)
#         yawn_avg = sum(yawns_queue) / len(yawns_queue)
#
#         # Calculating the drowsiness score
#         drowsiness_score = (blink_weight * blink_avg) + (yawn_weight * yawn_avg)
#
#         # Label image with regions of interest
#         image, _, _, _ = get_labeled_face(image, landmarks)
#
#         # log
#         print(f"Blink Avg: {blink_avg:.2f}, Yawn Avg: {yawn_avg:.2f}, Drowsiness Score: {drowsiness_score:.2f}")
#
#         # if past the warm-up period
#         if frame_count > warmup_frames:
#             # Check if the drowsiness score exceeds the threshold
#             if drowsiness_score > DROWSINESS_THRESHOLD:
#                 beep_thread = threading.Thread(target=beep)
#                 beep_thread.start()
#                 image = cv.putText(image, " --- DROWSY", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
#                                    color=(255, 1, 1))
#             else:
#                 image = cv.putText(image, " --- ALERT", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
#                                    color=(0, 255, 0))
#         else:
#             # warm-up period
#             image = cv.putText(image, " --- INITIALIZING", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
#                                color=(255, 255, 0))
#
#     else:
#         print("DRIVER ISN'T FACING FRONT!")
#         image = cv.putText(image, "DRIVER IS NOT FACING FRONT", (150, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
#                            color=(255, 1, 1))
#
#     cv.imshow("Results", image)
#     success, image = cap.read()
#
#     if cv.waitKey(delay_time) & 0xFF == ord("q"):
#         break
#
# cap.release()
# cv.destroyAllWindows()
#
#

import cv2 as cv
import dlib
import numpy as np


def get_frame(second):
    cap.set(cv.CAP_PROP_POS_MSEC, second * 1000)
    has_frames, img = cap.read()
    return has_frames, img


cap = cv.VideoCapture(r'/home/huda/Desktop/fyp files/FYP/test-sets/abc.mp4')
# cap.set(cv.CV_CAP_PROP_FPS, 60)
cap.set(3, 440)  # width
cap.set(4, 280)  # height
cap.set(10, 50)  # brightness

# read a frame from the video at time 1 second
success, frame = get_frame(1)
print(type(cap))
# verify that the frame was successfully read
assert success
assert isinstance(frame, np.ndarray)
assert frame.ndim == 3


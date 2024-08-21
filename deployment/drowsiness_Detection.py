import cv2 as cv
import dlib
import numpy as np
import threading
import os
from keras.models import save_model, load_model

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("detectors/shape_predictor_68_face_landmarks.dat")

yawning = load_model("models/yawn_model.h5")
eyes = load_model("models/blink_model.h5")


def get_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def get_facial_landmarks(gray):
    face_rectangles = detector(gray, 1)
    if len(face_rectangles) == 0:
        return -1
    face_landmarks = predictor(gray, face_rectangles[0])
    face_landmarks = np.array([[p.x, p.y] for p in face_landmarks.parts()])
    return face_landmarks


def beep():
    # os.system("echo -n '\a';sleep 0.2;" * x)
    os.system("beep -f 1000q -l 1500")


blinks = []
yawns = []

cap = cv.VideoCapture(r'test-sets/jkl.mp4')
# cap.set(cv.CV_CAP_PROP_FPS, 60)
cap.set(3, 440)  # width
cap.set(4, 280)  # height
cap.set(10, 50)  # brightness


def get_frame(second):
    cap.set(cv.CAP_PROP_POS_MSEC, second * 1000)
    has_frames, img = cap.read()
    return has_frames, img


# Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'mp4v') # or 'XVID'
# out = cv.VideoWriter('buffer.mp4', fourcc, 30.0, (640, 480)) # Change the dimensions and frame rate as per your requirement

images = []

sec = 0
frameRate = 0.5
fps = 300
delay_time = int(1000/fps)
# success,image = getFrame(sec)
success, image = cap.read()
per_close = 0


while success:

    sec = sec + frameRate
    sec = round(sec, 2)

    image = get_gray(image)
    s = get_facial_landmarks(image)
    if isinstance(s, np.ndarray):
        mx = s[48][0] - 5
        mxw = s[54][0] + 5
        my = s[51][1] - 7
        myh = s[57][1] + 8

        lx = s[18][0] - 8
        lxw = s[21][0] + 5
        ly = s[18][1] - 8
        lyh = s[41][1] + 8

        rx = s[22][0] - 8
        rxw = s[25][0] + 5
        ry = s[22][1] - 9
        ryh = s[46][1] + 5
        mouth = image[my:myh, mx:mxw]
        l_eye = image[ly:lyh, lx:lxw]
        r_eye = image[ry:ryh, rx:rxw]
        l_eye = cv.resize(l_eye, (256, 256))
        r_eye = cv.resize(r_eye, (256, 256))
        mouth = cv.resize(mouth, (256, 256))

        l_eye = np.expand_dims(l_eye, axis=0)
        r_eye = np.expand_dims(r_eye, axis=0)
        mouth = np.expand_dims(mouth, axis=0)

        blinks.append(eyes.predict(l_eye)[0][0])
        blinks.append(eyes.predict(r_eye)[0][0])
        yawns.append(yawning.predict(mouth)[0][0])

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
        print(f"Drowsy Level....{per_close:.2f}%")
        if per_close > 0.55:
            print(f"Drowsy Level....{per_close:.2f}%")
            beep_thread = threading.Thread(target=beep)
            beep_thread.start()
            image = cv.putText(image, " --- DROWSY", (50, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                               color=(255, 1, 1))
            # os.system("beep -f 1000q -l 1500")
    else:
        print("DRIVER ISN'T FACING FRONT!")
        image = cv.putText(image, "DRIVER IS NOT FACING FRONT", (150, 50), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                           color=(255, 1, 1))

    cv.imshow("Results", image)

    # success,image = getFrame(sec)
    success, image = cap.read()
        # get_frame(sec)

    if cv.waitKey(delay_time) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

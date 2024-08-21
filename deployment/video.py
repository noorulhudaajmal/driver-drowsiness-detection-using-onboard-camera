import cv2

# Read the video file
cap = cv2.VideoCapture(r'output_video.avi')

# Loop through each frame and display it
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

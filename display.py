import cv2
import numpy as np

# Web camera
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame1 = cap.read()

    cv2.imshow('DL Project Vehicle Counter', frame1)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release

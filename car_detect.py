import numpy as np
import cv2
from Training_vehicle_detection.run_sliding_window1 import run as detect_vehicle
def read(name):
    cap= cv2.VideoCapture(name)
    while cap.isOpened():
        ret, frame= cap.read()
        if ret == True:
            result, bbox = detect_vehicle(frame, debug=False)
            cv2.imshow('img', result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()

read('/Users/datle/Desktop/Official_license_plate/Traffic - 27260.mp4')



import cv2
import numpy as np

def color_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    if np.count_nonzero(green_mask) > np.count_nonzero(red_mask):
        return "Go"
    elif np.count_nonzero(red_mask) > np.count_nonzero(green_mask):
        return "Stop"
    else:
        return "Unknown"
    
'''
cap = cv2.VideoCapture("traffic light_3.mp4") # fix me

if not cap.isOpened():
    print("Error opening video file")

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 0)
        cv2.imshow('frame', frame)
        result = color_detection(frame)
        print(result)
    else:
        break
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
'''
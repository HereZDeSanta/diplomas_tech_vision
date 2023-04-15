import cv2

cap = cv2.VideoCapture('3.mp4')
if not cap.isOpened():
    print("Error opening video")

while(cap.isOpened()):
    status, frame = cap.read()
    if status:
        cv2.imshow('frame', frame)
    key = cv2.waitKey(500)

    if key == 32:
        cv2.waitKey()
    elif key == ord('q'):
        break
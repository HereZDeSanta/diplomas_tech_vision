import cv2
import numpy as np

cap = cv2.VideoCapture('3.mp4')
percent = 100

roi_x = 330
roi_y = 430
roi_width = 90
roi_height = 110

red_lower = np.array([0, 137, 249], dtype="uint8")
red_upper = np.array([15, 255, 255], dtype="uint8")
yellow_lower = np.array([17, 165, 130], dtype="uint8")
yellow_upper = np.array([101, 255, 255], dtype="uint8")
green_lower = np.array([40, 85, 180], dtype="uint8")
green_upper = np.array([91, 255, 255], dtype="uint8")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #frame = cv2.resize(frame, (720, 480))

    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    brightness = lab[:, :, 0]

    brightness_mask = cv2.threshold(brightness, 50, 255, cv2.THRESH_BINARY)[1]

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_weighted_sum = np.sum(cv2.bitwise_and(red_mask, brightness_mask) * brightness)
    yellow_weighted_sum = np.sum(cv2.bitwise_and(yellow_mask, brightness_mask) * brightness)
    green_weighted_sum = np.sum(cv2.bitwise_and(green_mask, brightness_mask) * brightness)

    if red_weighted_sum > yellow_weighted_sum and red_weighted_sum > green_weighted_sum:
        color = 'red'
    elif yellow_weighted_sum > red_weighted_sum and yellow_weighted_sum > green_weighted_sum:
        color = 'yellow'
    elif green_weighted_sum > red_weighted_sum and green_weighted_sum > yellow_weighted_sum:
        color = 'green'
    else:
        color = 'none'

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
    cv2.putText(frame, color, (roi_x, roi_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imshow("Traffic Light Detection from Video", frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

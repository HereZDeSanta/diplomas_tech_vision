import cv2
import numpy as np


red_lower = ([0, 0, 200], dtype="uint8")
red_upper = np.array([15, 50, 255], dtype="uint8")
yellow_lower = np.array ([20, 100, 100], dtype="uint8")
yellow_upper = np.array([30, 255, 255], dtype="uint8")
green_lower = np.array([50, 100, 100], dtype="uint8")
green_upper = np.array([70, 255, 255], dtype="uint8")

# Initialize the camera
cap = cv2.VideoCapture('3.mp4')

while True:
    # Read a frame from the camera
    _, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color mask
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    mask_full = red_mask + yellow_mask + green_mask
    cv2.imshow("mask_hsv", mask_full)

    # Find the contours of the colored regions
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a circle around the colored regions
    for contour in red_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.circle(frame, (int(x + w/2), int(y + h/2)), int(w/2), (0, 0, 255), 2)
    for contour in yellow_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.circle(frame, (int(x + w/2), int(y + h/2)), int(w/2), (0, 255, 255), 2)
    for contour in green_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.circle(frame, (int(x + w/2), int(y + h/2)), int(w/2), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Traffic lights", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close the window
cv2.destroyAllWindows()
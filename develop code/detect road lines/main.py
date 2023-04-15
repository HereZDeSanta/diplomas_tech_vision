# import cv2
# import numpy as np
#
# # Load the video file
# cap = cv2.VideoCapture('test_road.mp4')
#
# # Define the color threshold ranges for white road markings in HSV color space
# white_lower = np.array([10, 0, 80])
# white_upper = np.array([255,50, 255])
#
# # Define the kernel for morphological operations
# kernel = np.ones((5,5), np.uint8)
#
# # Define the parameters for Canny edge detection
# canny_threshold1 = 10
# canny_threshold2 = 150
#
# # Define the parameters for Hough line detection
# rho = 1
# theta = np.pi / 180
# threshold = 10
# min_line_length = 20
# max_line_gap = 5
#
# # Process each frame of the video
# while cap.isOpened():
#     # Read the current frame
#     ret, frame = cap.read()
#
#     # Convert the frame to HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Threshold the image to detect white road markings
#     white_mask = cv2.inRange(hsv, white_lower, white_upper)
#
#     # Apply morphological operations to the binary mask to remove noise
#     mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#
#     # Apply Canny edge detection to the binary mask
#     edges = cv2.Canny(mask, canny_threshold1, canny_threshold2)
#
#     # Apply Hough line detection to the edge map
#     lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
#
#     # Draw the detected lines on the frame
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Road Marking Detection', frame)
#
#     # Check if the user wants to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video file and close the window
# cap.release()
# cv2.destroyAllWindows()


#тест 1

import cv2
import numpy as np
import lanes

cap = cv2.VideoCapture("test_road.mp4")

if not cap.isOpened():
    print("Error opening video")


while cap.isOpened():
    _, frame = cap.read()
    copy_frame = np.copy(frame)

    try:
        frame = lanes.canny(frame)
        frame = lanes.ROI(frame)
        detect = cv2.HoughLinesP(frame, 2, np.pi/180, 100, np.array([()]), minLineLength=20, maxLineGap=5)
        averaged_lines = lanes.average_slope_intercept(frame, detect)
        line_detect = lanes.display_lines(copy_frame, averaged_lines)
        final_detect = cv2.addWeighted(copy_frame, 0.8, line_detect, 0.8, 1)
        cv2.imshow("Road Detect", final_detect)
        cv2.imshow("ROI Detect", frame)
    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
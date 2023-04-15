# import numpy as np
# import cv2
#
# percent = 60
# cap = cv2.VideoCapture('test3.mp4')
# if not cap.isOpened():
#     print("Error opening video")
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     width = int(frame.shape[1] * percent / 100)
#     height = int(frame.shape[0] * percent / 100)
#     dim = (width, height)
#     frame_re = cv2.resize(frame, dim)
#
#     cv2.imshow("video_frame", frame_re)
#     kernel = np.ones((5, 5), np.uint8)
#     frame_hsv = cv2.cvtColor(frame_re, cv2.COLOR_BGR2HSV)
#
#     lower_red = np.array([0, 137, 249], dtype="uint8")
#     upper_red = np.array([15, 255, 255], dtype="uint8")
#     lower_yellow = np.array([17, 165, 130], dtype="uint8")
#     upper_yellow = np.array([101, 255, 255], dtype="uint8")
#     lower_green = np.array([40, 85, 180], dtype="uint8")
#     upper_green = np.array([91, 255, 255], dtype="uint8")
#
#     mask_red = cv2.inRange(frame_hsv, lower_red, upper_red)
#     mask_yellow = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
#     mask_green = cv2.inRange(frame_hsv, lower_green, upper_green)
#
#     mask_full = mask_red + mask_yellow + mask_green
#     mask_full = cv2.bitwise_and(frame_re, frame_re, mask=mask_full)
#     cv2.imshow("mask_hsv", mask_full)
#
#     pic3 = cv2.GaussianBlur(mask_full, (13, 13), 4)
#     edges = cv2.Canny(pic3, 130, 85, 7)
#     pic1 = cv2.dilate(edges, kernel, iterations=4)
#     cv2.imshow("mask_dilation", pic1)
#     a = frame_re.shape[0]
#     b = frame_re.shape[1]
#     print([a,b])
#     contours, hierarcy = cv2.findContours(pic1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for i in range(len(contours)):
#         (x, y), radius = cv2.minEnclosingCircle(contours[i])
#         if x < 400 and y < 500:
#             if cv2.contourArea(contours[i]) > 100:
#                 con = cv2.drawContours(frame_re, contours, i, [255, 0, 139], 2)
#                 stop = 1
#                 cv2.imshow('contours1', con)
#             else:
#                 stop = 0
#                 cv2.imshow('contours3', frame_re)
#             print(stop)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()



# Область ROI
import cv2
import numpy as np

# Load video file or webcam
cap = cv2.VideoCapture('test3.mp4')
percent = 70
# Define ROI coordinates
roi_x = 440
roi_y = 400
roi_width = 90
roi_height = 120

# Define color ranges for red, yellow, and green traffic lights in the HSV color space
red_lower = np.array([0, 137, 249])
red_upper = np.array([15, 255, 255])
yellow_lower = np.array([17, 165, 130])
yellow_upper = np.array([101, 255, 255])
green_lower = np.array([40, 85, 180])
green_upper = np.array([91, 255, 255])

while True:
    # Read frame from video file or webcam
    ret, frame = cap.read()
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim)

    # If frame cannot be read, break loop
    if not ret:
        break

    # Crop the frame to the ROI
    roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    # Convert the ROI to the HSV color space
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Threshold the ROI based on the red, yellow, and green color ranges
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # Combine the red, yellow, and green masks
    mask = cv2.bitwise_or(red_mask, yellow_mask)
    mask = cv2.bitwise_or(mask, green_mask)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a rectangle around the ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)

    # Iterate through the contours
    for contour in contours:
        # Calculate the area and centroid of each contour
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        centroid_x = int(x + w/2)
        centroid_y = int(y + h/2)

        # Ignore small contours
        if area < 100:
            continue

        # Determine the color of the contour based on its position in the ROI and its average color in the HSV color space
        # if centroid_x < roi_width/3:
        #     color = 'red'
        #     avg_color = np.mean(hsv_frame[y:y+h, x:x+w], axis=(0,1))
        # elif centroid_x < roi_width*1/3:
        #     color = 'yellow'
        #     avg_color = np.mean(hsv_frame[y:y+h, x:x+w], axis=(0,1))
        # else:
        #     color = 'green'
        #     avg_color = np.mean(hsv_frame[y:y+h, x:x+w], axis=(0,1))

        # Draw a rectangle and label for each traffic light
        cv2.rectangle(frame, (roi_x+x, roi_y+y), (roi_x+x+w, roi_y+y+h), (0, 0, 255), 2)
        # Draw label for each traffic light
        # cv2.putText(frame, color, (roi_x + x, roi_y + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Traffic Light Detector', frame)
    cv2.imshow('ROI', roi_frame)
    cv2.imshow('HSV', mask)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# import numpy as np
# import cv2
# import math as m
#
# cap = cv2.VideoCapture('test1.mp4')
# if not cap.isOpened():
#     print("Error opening video")
#
# #Кусок кода для воспроизведения видео по кадрам (начиная с if ret)
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     # if ret:
#     #     cv2.imshow('frame', frame)
#     # key = cv2.waitKey(500)
#     #
#     # if key == 32:
#     #     cv2.waitKey()
#     # elif key == ord('q'):
#     #     break
#
#     cv2.imshow("video_frame", frame)
#     kernel = np.ones((3, 3), np.uint8)
#     frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
#     mask_full = cv2.bitwise_and(frame, frame, mask=mask_full)
#     # contours, hierarcy = cv2.findContours(red_mask_full, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cv2.imshow("mask_hsv", mask_full)
#     # pic_erode = cv2.erode(mask_full, kernel, iterations=1)
#     pic3 = cv2.GaussianBlur(mask_full, (13, 13), 4)
#     edges = cv2.Canny(pic3, 130, 85, 7)
#     pic1 = cv2.dilate(edges, kernel, iterations=4)
#     cv2.imshow("mask_dilation", pic1)
#     a = frame.shape[0]
#     b=frame.shape[1]
#     print([a,b])
#     contours, hierarcy = cv2.findContours(pic1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for i in range(len(contours)):
#         (x, y), radius = cv2.minEnclosingCircle(contours[i])
#         # print(x,y)
#         if x < 400 and y < 500:
#             if cv2.contourArea(contours[i]) > 100:
#                 con = cv2.drawContours(frame, contours, i, [255, 0, 139], 2)
#                 # cv2.putText(frame,'STOP',(50,90),cv2.FONT_HERSHEY_COMPLEX,3,[0,0,255],5)
#                 stop = 1
#                 cv2.imshow('contours2', con)
#             else:
#                 stop = 0
#                 # cv2.putText(frame,'GO',(50,90),cv2.FONT_HERSHEY_COMPLEX,3,[0,0,255],5)
#                 cv2.imshow('contours2', frame)
#             print(stop)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()




# print(contours)
# for i in range(len(contours)):
#      # con = cv2.drawContours(frame,contours,i,[0,255,0], 3)
#      (x,y),radius = cv2.minEnclosingCircle(contours[i])
#      print('x',x)
#      print('y',y)
#      circle_square=m.pi*radius*radius
#      # cv2.circle(frame,(int(x),int(y)),int(radius),[255,255,0],3)
#      area = cv2.contourArea(contours[i])
#      if radius >=20:
#         if abs(area-circle_square)<=300 and abs(area-circle_square)>=5:
#             maybe_red=contours[i]
#             con2 = cv2.drawContours(frame,contours,i,[255,0,0], 3)
#             cv2.imshow('contours2', con2)
#
# cv2.imshow('contours',red_mask_full)
# cv2.waitKey(0)
# cv2.imshow("rrr", red_mask_full)
# a = red_mask_full.shape[0]
# circles = cv2.HoughCircles(red_mask_full, cv2.HOUGH_GRADIENT, 1, minDist = a//40, param1 = 10, param2 = 5, minRadius = a//1000, maxRadius = a//2)
# print(circles[0])
# for circle in circles[0]:
#     center = int(circle[0]), int(circle[1])
#     radius = int(circle[2])
# cv2.circle(frame, center, radius, (0, 0, 255), 3)
# cv2.imshow("rrr", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#
#
#
# Для поиска пространства HSV
# import cv2
# import numpy as np

# def nothing(x):
#     pass
#
#
# # red_lower = np.array([0, 150, 220], dtype='uint8')
# # red_upper = np.array([15, 255, 255], dtype='uint8')
# # yellow_lower = np.array([25, 100, 180], dtype='uint8')
# # yellow_upper = np.array([32, 255, 255], dtype='uint8')
# # green_lower = np.array([40, 85, 180], dtype='uint8')
# # green_upper = np.array([91, 255, 255], dtype='uint8')
#
# # cv2.namedWindow("HSV")
# # cv2.createTrackbar("Hue", "HSV", 0, 255, nothing)
# # cv2.createTrackbar("Saturation", "HSV", 0, 255, nothing)
# # cv2.createTrackbar("Value", "HSV", 0, 255, nothing)
#
# cap = cv2.VideoCapture('3.mp4')
# if not cap.isOpened():
#     print("Error opening video")
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     # if ret:
#     #     cv2.imshow('frame', frame)
#     # key = cv2.waitKey(500)
#     #
#     # if key == 32:
#     #     cv2.waitKey()
#     # elif key == ord('q'):
#     #     break
#     kernel = np.ones((3, 3), np.uint8)
#
#
#     # H = cv2.getTrackbarPos("Hue", "HSV")
#     # S = cv2.getTrackbarPos("Saturation", "HSV")
#     # V = cv2.getTrackbarPos("Value", "HSV")
#
#     red_lower = np.array([0, 137, 249], dtype='uint8')
#     red_upper = np.array([15, 255, 255], dtype='uint8')
#     yellow_lower = np.array([17, 165, 130], dtype='uint8')
#     yellow_upper = np.array([101, 255, 255], dtype='uint8')
#     green_lower = np.array([40, 85, 180], dtype='uint8')
#     green_upper = np.array([91, 255, 255], dtype='uint8')
#
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     red_mask = cv2.inRange(hsv, red_lower, red_upper)
#     yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
#     green_mask = cv2.inRange(hsv, green_lower, green_upper)
#     mask_full = red_mask + yellow_mask + green_mask
#     # cv2.imshow("mask_hsv", mask_full)
#     cv2.imshow('yellow', yellow_mask)
#
#     pic3 = cv2.GaussianBlur(mask_full, (9, 9), 4)
#     edges = cv2.Canny(pic3, 130, 85, 7)
#     pic1 = cv2.dilate(edges, kernel, iterations=4)
#     cv2.imshow("mask_dilation", pic1)
#
#
#     red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
#     for contour in red_contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.circle(frame, (int(x + w/2), int(y + h/2)), int(w/2), (0, 0, 255), 2)
#     for contour in yellow_contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.circle(frame, (int(x + w/2), int(y + h/2)), int(w/2), (0, 255, 255), 2)
#     for contour in green_contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.circle(frame, (int(x + w/2), int(y + h/2)), int(w/2), (0, 255, 0), 2)
#     cv2.imshow("Traffic lights", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()





#ROI область

# import cv2
#
# # Define the range of colors for the traffic lights
# red_lower = (0, 0, 200)
# red_upper = (50, 50, 255)
# yellow_lower = (20, 100, 100)
# yellow_upper = (30, 255, 255)
# green_lower = (50, 100, 100)
# green_upper = (70, 255, 255)
#
# # Define the region of interest (ROI)
# roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 200
#
# # Initialize the camera
# cap = cv2.VideoCapture('3.mp4')
#
# while True:
#     # Read a frame from the camera
#     _, frame = cap.read()
#
#     # Define the region of interest (ROI)
#     roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
#
#     # Convert the frame to HSV color space
#     hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#
#     # Define the color mask
#     red_mask = cv2.inRange(hsv, red_lower, red_upper)
#     yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
#     green_mask = cv2.inRange(hsv, green_lower, green_upper)
#
#     # Find the contours of the colored regions
#     red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Draw a circle around the colored regions
#     for contour in red_contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.circle(roi, (x + int(w/2), y + int(h/2)), int(w/2), (0, 0, 255), 2)
#     for contour in yellow_contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.circle(roi, (x + int(w/2), y + int(h/2)), int(w/2), (0, 255, 255), 2)
#     for contour in green_contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.circle(roi, (x + int(w / 2), y + int(h / 2)), int(w / 2), (0, 255, 0), 2)
#         # Add the ROI back to the original frame
#         frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi
#
#         # Show the frame
#         cv2.imshow("Traffic lights", frame)
#
#         # Exit the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()





















import numpy as np
import cv2

percent = 100
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error opening video")

while (cap.isOpened()):
    ret, frame = cap.read()
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    frame_re = cv2.resize(frame, dim)
    cv2.imshow("video_frame", frame_re)
    kernel = np.ones((3, 3), np.uint8)
    frame_hsv = cv2.cvtColor(frame_re, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 137, 249], dtype="uint8")
    upper_red = np.array([15, 255, 255], dtype="uint8")
    lower_yellow = np.array([17, 165, 130], dtype="uint8")
    upper_yellow = np.array([101, 255, 255], dtype="uint8")
    lower_green = np.array([40, 85, 180], dtype="uint8")
    upper_green = np.array([91, 255, 255], dtype="uint8")

    mask_red = cv2.inRange(frame_hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(frame_hsv, lower_green, upper_green)

    mask_full = mask_red + mask_yellow + mask_green
    mask_full = cv2.bitwise_and(frame_re, frame_re, mask=mask_full)
    cv2.imshow("mask_hsv", mask_full)

    pic3 = cv2.GaussianBlur(mask_full, (13, 13), 4)
    edges = cv2.Canny(pic3, 130, 85, 7)
    pic1 = cv2.dilate(edges, kernel, iterations=4)
    cv2.imshow("mask_dilation", pic1)
    a = frame_re.shape[0]
    b = frame_re.shape[1]
    print([a,b])
    contours, hierarcy = cv2.findContours(pic1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        (x, y), radius = cv2.minEnclosingCircle(contours[i])
        if x < 400 and y < 500:
            if cv2.contourArea(contours[i]) > 100:
                con = cv2.drawContours(frame_re, contours, i, [255, 0, 139], 2)
                stop = 1
                cv2.imshow('contours1', con)
            else:
                stop = 0
                cv2.imshow('contours3', frame_re)
            print(stop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



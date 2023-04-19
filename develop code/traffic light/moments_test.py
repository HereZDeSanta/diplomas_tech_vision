# Хочу основой сделать
import cv2
import numpy as np
import math

# Load image and convert to grayscale
img = cv2.imread('traffic_light.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to segment the traffic lights
thresh = cv2.threshold(gray, 70,255, cv2.THRESH_BINARY)[1]

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define empty arrays to store circle center coordinates and radii
circle_arr = []
radius_arr = []

# Loop through contours
for c in contours:
    M = cv2.moments(c)
    epsilon = 0.009 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    n = approx.ravel()
#     i = 0
#
    # If contour moment is non-zero
    if M["m00"] != 0:
        # Calculate contour center and area
        cX = int(M["m10"]/ M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = cv2.contourArea(c)
#
        # If area is greater than or equal to 900
        if area >= 900 and len(approx) > 10:
                cir_string = str(cX) + "," + str(cY)
                circle_arr.append(cir_string)
                area = cv2.contourArea(c)
                circle_radius = math.sqrt(area / math.pi)
                radius_arr.append(circle_radius)
                cv2.circle(img, (cX, cY), 3, (0, 255, 0), 3)
        # else:
        #         cX, cY = 0, 0

    cv2.imshow('Contours', img)
#
# # Draw circles on image
if len(radius_arr) >= 3:
    for i in range(len(circle_arr)):
        cX, cY = map(int, circle_arr[i].split(','))
        radius = int(radius_arr[i])
        cv2.circle(img, (cX, cY), radius, (0, 255, 0), 3)

cv2.imshow('Final Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()






# Залупа на воротнике
# import cv2
# import numpy as np
# import math
#
# # Load image and convert to grayscale
# image = cv2.imread('traffic_light.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply thresholding to segment the traffic lights
# thresh = cv2.threshold(gray, 70,255, cv2.THRESH_BINARY)[1]
#
# # Find contours
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Define empty arrays to store circle center coordinates and radii
# circle_arr = []
# radius_arr = []
#
# # Loop through contours
# for c in contours:
#     M = cv2.moments(c)
#     epsilon = 0.009 * cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, epsilon, True)
#
#     # If contour moment is non-zero
#     if M["m00"] != 0:
#         # Calculate contour center and area
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         area = cv2.contourArea(c)
#
#         # If area is greater than or equal to 900
#         if area >= 900:
#             # Draw contour and center point
#             # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#             cv2.circle(image, (cX, cY), 3, (255, 255, 255), -1)
#
#             # If contour has more than 10 sides and is not a triangle or quadrilateral
#             if len(approx) != 3 and len(approx) != 4 and len(approx) > 15:
#                 # Calculate circle center coordinates and radius
#                 cir_string = str(cX) + "," + str(cY)
#                 circle_arr.append(cir_string)
#                 circle_area = cv2.contourArea(c)
#                 circle_radius = math.sqrt(circle_area / math.pi)
#
#                 # If radius is not already in list, add it
#                 if circle_radius not in radius_arr:
#                     radius_arr.append(circle_radius)
#
#     cv2.imshow('Contours', image)
#
#     # Break out of loop if three unique radii have been found
#     if len(radius_arr) == 3:
#         if radius_arr[0] == radius_arr[1] == radius_arr[2]:
#             print("Found three identical radii!")
#             break
#
# # Draw circles on image
# for i in range(len(circle_arr)):
#     cX, cY = map(int, circle_arr[i].split(','))
#     radius = int(radius_arr[i])
#     cv2.circle(image, (cX, cY), radius, (0, 0, 255), 2)
#
# # Show final image
# cv2.imshow('Final Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




#Ебучая имба
# import cv2
# import numpy as np
# import math
#
# # Load image and convert to grayscale
# img = cv2.imread('traffic_light.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Apply thresholding to segment the traffic lights
# thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]
#
# # Find contours
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # Define empty arrays to store circle center coordinates and radii
# circle_arr = []
# radius_arr = []
#
# # Loop through contours
# for c in contours:
#     M = cv2.moments(c)
#     epsilon = 0.009 * cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, epsilon, True)
#     n = approx.ravel()
#
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         area = cv2.contourArea(c)
#
#         # Only consider contours with area >= 900 and number of sides > 15
#         if area >= 900 and len(approx) > 15:
#             # Store circle center coordinates and radius
#             cir_string = str(cX) + "," + str(cY)
#             circle_arr.append(cir_string)
#             area = cv2.contourArea(c)
#             circle_radius = math.sqrt(area / math.pi)
#             radius_arr.append(circle_radius)
#
#             # Draw contour and center point
#             # cv2.drawContours(img, [c], 0, (0, 255, 0), 2)
#             cv2.circle(img, (cX, cY), 3, (255, 255, 255), -1)
#
# # Find circles with three identical radii
# if len(radius_arr) >= 3:
#     for i in range(len(circle_arr)):
#         cX, cY = map(int, circle_arr[i].split(','))
#         radius = int(radius_arr[i])
#         cv2.circle(img, (cX, cY), radius, (0, 255, 0), 3)
#
#         #Кусок, ятобы обводить каждый сигнал разным цветом
#     # for i in range(len(radius_arr)):
#     #     for j in range(i+1, len(radius_arr)):
#     #         for k in range(j+1, len(radius_arr)):
#     #             if abs(radius_arr[i] - radius_arr[j]) < 5 and abs(radius_arr[j] - radius_arr[k]) < 5:
#     #                 cX1, cY1 = map(int, circle_arr[k].split(','))
#     #                 cX2, cY2 = map(int, circle_arr[j].split(','))
#     #                 cX3, cY3 = map(int, circle_arr[i].split(','))
#     #                 cv2.circle(img, (cX1, cY1), int(radius_arr[i]), (0, 0, 255), 3)
#     #                 cv2.circle(img, (cX2, cY2), int(radius_arr[j]), (0, 165, 255), 3)
#     #                 cv2.circle(img, (cX3, cY3), int(radius_arr[k]), (0, 255, 0), 3)
#
# cv2.imshow('Final Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



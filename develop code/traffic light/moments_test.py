#ПЕРВЫЙ ТЕСТ ПРОГИ(ЗАБРАКОВАН)

# import cv2
# import numpy as np
# import math
#
#
# img = cv2.imread('traffic_light.jpg', cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 70,255, cv2.THRESH_BINARY)[1]
# cv2.imshow("binary_image", thresh)
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# circle_arr = []
# radius_arr = []
#
#
# for c in contours:
#     M = cv2.moments(c)
#     epsilon = 0.009 * cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, epsilon, True)
#     # n = approx.ravel()
#     # i = 0
#
#     if M["m00"] != 0:
#         cX = int(M["m10"]/ M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         area = cv2.contourArea(c)
#
#         if area >= 900 and len(approx) > 15: #для ravel (n), значение - >32
#                 cir_string = str(cX) + "," + str(cY)
#                 circle_arr.append(cir_string)
#                 area = cv2.contourArea(c)
#                 circle_radius = math.sqrt(area / math.pi)
#                 radius_arr.append(circle_radius)
#                 cv2.drawContours(img, [c], 0, (0, 255, 0), 2)
#                 cv2.circle(img, (cX, cY), 2, (255, 0, 139), 3)
#                 print(f"Circle Center: ({cX}, {cY}) | Radius: {circle_radius}")
#         else:
#                 cX, cY = 0, 0
#
#     cv2.imshow('Contours', img)
#
# if len(radius_arr) >= 3:
#     for i in range(len(circle_arr)):
#         cX, cY = map(int, circle_arr[i].split(','))
#         radius = int(radius_arr[i])
#         #Обведем сигналы светофора цветом влюбленной жабы
#         cv2.circle(img, (cX, cY), radius, (60, 170, 60), 3)
#
# #         #Кусок, ятобы обводить каждый сигнал разным цветом
# #     # for i in range(len(radius_arr)):
# #     #     for j in range(i+1, len(radius_arr)):
# #     #         for k in range(j+1, len(radius_arr)):
# #     #             if abs(radius_arr[i] - radius_arr[j]) < 5 and abs(radius_arr[j] - radius_arr[k]) < 5:
# #     #                 cX1, cY1 = map(int, circle_arr[k].split(','))
# #     #                 cX2, cY2 = map(int, circle_arr[j].split(','))
# #     #                 cX3, cY3 = map(int, circle_arr[i].split(','))
# #     #                 cv2.circle(img, (cX1, cY1), int(radius_arr[i]), (0, 0, 255), 3)
# #     #                 cv2.circle(img, (cX2, cY2), int(radius_arr[j]), (0, 165, 255), 3)
# #     #                 cv2.circle(img, (cX3, cY3), int(radius_arr[k]), (0, 255, 0), 3)
#
#
# cv2.imshow('Detect lights', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# #Вроде как то, что просила Катерина
import cv2
import numpy as np
import math

img = cv2.imread('Traf1.png')

roi_x1, roi_y1 = 220, 10
roi_x2, roi_y2 = 350, 250
cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
roi = img[roi_y1:roi_y2, roi_x1:roi_x2]

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (13, 13), 0)
canny = cv2.Canny(blur, 50, 80, 5)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 4)
cv2.imshow("binary_image", thresh)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

circle_arr = []
radius_arr = []

for i in range(len(contours)):
    c = contours[i]
    M = cv2.moments(c)
    epsilon = 0.009 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = cv2.contourArea(c)

        if area >= 900 and len(approx) > 15:
            circle_arr.append((cX, cY))
            area = cv2.contourArea(c)
            circle_radius = math.sqrt(area / math.pi)
            radius_arr.append(circle_radius)
            cv2.drawContours(img, [c], 0, (0, 255, 0), 2)
            cv2.circle(img, (cX, cY), 2, (255, 0, 139), 3)
            print(f"Circle Center: ({cX}, {cY}) | Radius: {circle_radius}")


if len(radius_arr) == 3:
    circle_arr.sort(key=lambda x: x[0])

    for i in range(len(circle_arr) - 2):
        c1_x, c1_y = circle_arr[i]
        c2_x, c2_y = circle_arr[i + 1]
        c3_x, c3_y = circle_arr[i + 2]


        if abs(radius_arr[i] - radius_arr[i + 1]) < 5 and abs(radius_arr[i + 1] - radius_arr[i + 2]) < 5 \
                and abs(c2_x - c1_x - (c3_x - c2_x)) < 5 and abs(c2_y - c1_y - (c3_y - c2_y)) < 5 \
                and abs(cv2.contourArea(contours[i]) - cv2.contourArea(contours[i + 1])) < 900 \
                and abs(cv2.contourArea(contours[i + 1]) - cv2.contourArea(contours[i + 2])) < 900:

            rect_center_x = int((c1_x + c2_x + c3_x) / 3)
            rect_center_y = int((c1_y + c2_y + c3_y) / 3)

            circle_distance = int(abs(c2_x - c1_x))
            rect_width = int(circle_distance + max(radius_arr[i], radius_arr[i + 1], radius_arr[i + 2]) *2)
            rect_height = int(max(radius_arr[i], radius_arr[i + 1], radius_arr[i + 2]) * 2)

            cv2.rectangle(img, (rect_center_x - rect_width, rect_center_y - rect_height),
                          (rect_center_x + rect_width, rect_center_y + rect_height),
                          (0, 0, 255), 2)

cv2.imshow("image traffic_light_detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()






#ЧЕРЕЗ ХАФА(ВИДИМО ПРОБЛЕМА С ПАРАМЕТРАМИ)
# import cv2
# import numpy as np
# import math
#
#
# img = cv2.imread('traffic_light.jpg', cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# thresh = cv2.threshold(gray, 42, 255, cv2.THRESH_BINARY_INV)[1]
# cv2.imshow("binary_image", thresh)
#
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# traffic_lights_arr = []
#
# for c in contours:
#     M = cv2.moments(c)
#
#
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#
#         area = cv2.contourArea(c)
#
#
#         if area >= 900 and len(c) > 15:
#             circle_radius = int(math.sqrt(area / math.pi))
#
#             found_traffic_light = False
#             for x, y, radius in traffic_lights_arr:
#                 if abs(x - cX) < 5 and abs(y - cY) < 5 and abs(radius - circle_radius) < 900:
#                     found_traffic_light = True
#                     break
#
#             if not found_traffic_light:
#                 mask = np.zeros_like(thresh)
#                 cv2.drawContours(mask, [c],0, 255, -1)
#                 mask = cv2.bitwise_and(thresh, mask)
#                 cv2.imshow("mask", mask)
#
#                 circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=20,
#                                            minRadius=circle_radius - 5, maxRadius=circle_radius + 5)
#
#                 if circles is not None and len(circles) == 3:
#                     traffic_lights_arr.append((cX, cY, circle_radius))
#
#
#                     for circle in circles:
#                         x, y, radius = circle[0]
#                         cv2.circle(img, (int(x), int(y)), int(radius), (0, 0, 255), 3)
#
# cv2.imshow('Detect lights', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()










# #Через область ROI
# import cv2
# import numpy as np
# import math
#
# img = cv2.imread('Traf5.png', cv2.IMREAD_COLOR)
#
#
# roi_x, roi_y, roi_w, roi_h = 770, 250, 120, 320
# roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
#
# gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("Binary Image", thresh)
#
#
# cv2.rectangle(img, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)
# cv2.imshow("Original Image with ROI", img)
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# circle_arr = []
# radius_arr = []
#
# for i in range(len(contours)):
#     c = contours[i]
#     M = cv2.moments(c)
#     epsilon = 0.009 * cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, epsilon, True)
#
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         area = cv2.contourArea(c)
#
#         if area >= 900 and len(approx) > 15:
#             circle_arr.append((cX, cY))
#             area = cv2.contourArea(c)
#             circle_radius = math.sqrt(area / math.pi)
#             radius_arr.append(circle_radius)
#             cv2.drawContours(roi, [c], 0, (0, 255, 0), 2)
#             print(f"Circle Center: ({cX}, {cY}) | Radius: {circle_radius}")
#
#
# if len(radius_arr) >= 3:
#     circle_arr.sort(key=lambda x: x[0])
#
#
#     for i in range(len(circle_arr) - 2):
#         c1_x, c1_y = circle_arr[i]
#         c2_x, c2_y = circle_arr[i + 1]
#         c3_x, c3_y = circle_arr[i + 2]
#
#
#         if abs(radius_arr[i] - radius_arr[i + 1]) < 5 and abs(radius_arr[i + 1] - radius_arr[i + 2]) < 5 \
#                 and abs(c2_x - c1_x - (c3_x - c2_x)) < 5 and abs(c2_y - c1_y - (c3_y - c2_y)) < 5 \
#                 and abs(cv2.contourArea(contours[i]) - cv2.contourArea(contours[i + 1])) < 900 \
#                 and abs(cv2.contourArea(contours[i + 1]) - cv2.contourArea(contours[i + 2])) < 900:
#
#             cv2.circle(roi, (int((c1_x + c3_x) / 2), int((c1_y + c3_y) / 2)), 10, (0, 255, 0), 2)
#             light = img[roi_y + c1_y - 70:roi_y + c3_y + 70, roi_x + c1_x - 70:roi_x + c3_x + 70]
#             cv2.imshow("Detected Traffic Light", light)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()







# #Фильтр Калмана(НУ ЭТО ВИДИМО КОНКРЕТНО МИМО)
# import cv2
# import numpy as np
#
# # Define traffic light colors (in BGR format)
# RED = (0, 0, 255)
# GREEN = (0, 255, 0)
#
# # Define initial search boxes (set manually). These will be dynamically updated.
# red_box = (307, 16, 10, 10)
# green_box = (308, 45, 10, 10)
#
# # Define sliding window parameters
# window_size = (20, 20)
# stride = 5
#
# # Define Kalman filter parameters
# state_size = 4
# measurement_size = 2
# kf = cv2.KalmanFilter(state_size, measurement_size)
# kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
# kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
# kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]], np.float32)
# kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32)
#
# for im_num in range(1, 700):
#     # Load image
#     img = cv2.imread('Traf2.png', cv2.IMREAD_COLOR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Predict position of traffic light using Kalman filter
#     pred = kf.predict()
#     x, y = int(pred[0]), int(pred[1])
#
#     # Define search window based on Kalman filter prediction
#     x1, y1 = x - window_size[0] // 2, y - window_size[1] // 2
#     x2, y2 = x1 + window_size[0], y1 + window_size[1]
#
#     # Search for red and green traffic lights in search window
#     red_mask = cv2.inRange(img[y1:y2, x1:x2], np.array([0, 0, 100]), np.array([100, 100, 255]))
#     green_mask = cv2.inRange(img[y1:y2, x1:x2], np.array([0, 100, 0]), np.array([100, 255, 100]))
#
#     # Find best match for red and green lights
#     best_match = None
#     best_match_val = 0
#     for y_start in range(y1, y2 - window_size[1] + 1, stride):
#         for x_start in range(x1, x2 - window_size[0] + 1, stride):
#             red_match_val = cv2.matchTemplate(red_mask[y_start - y1:y_start - y1 + window_size[1], x_start - x1:x_start - x1 + window_size[0]], np.ones(window_size), cv2.TM_CCORR_NORMED)
#             green_match_val = cv2.matchTemplate(green_mask[y_start - y1:y_start - y1 + window_size[1], x_start - x1:x_start - x1 + window_size[0]], np.ones(window_size), cv2.cv2.TM_CCORR_NORMED)
#             if red_match_val > best_match_val:
#                 best_match_val = red_match_val
#             best_match = (x_start, y_start, RED)
#             if green_match_val > best_match_val:
#                 best_match_val = green_match_val
#             best_match = (x_start, y_start, GREEN)
#             # Update Kalman filter with measurement
#             if best_match is not None:
#                 measurement = np.array([best_match[0] + window_size[0] // 2, best_match[1] + window_size[1] // 2],
#                                        np.float32)
#                 kf.correct(measurement)
#
#             # Draw search box and predicted traffic light position on image
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.circle(img, (int(kf.state[0]), int(kf.state[1])), 5, (0, 0, 255), -1)
#
#             # Draw traffic light detection box and color on image
#             if best_match is not None:
#                 cv2.rectangle(img, (best_match[0], best_match[1]),
#                               (best_match[0] + window_size[0], best_match[1] + window_size[1]), best_match[2], 2)
#
#             # Show image and wait for key press
#             cv2.imshow('image', img)
#             k = cv2.waitKey(1)
#
#             # If 'q' is pressed, exit loop
#             if k == ord('q'):
#                 break
#
# cv2.destroyAllWindows()






















#ТЕСТ ГИТОВСКОГО КОДА(НЕ СИЛЬНО В НЕМ УСПЕЛ РАЗОБРАТЬСЯ, НО ОН ЧЕТ НЕ ЗАПУСТИЛСЯ)
# import cv2
# import numpy as np
#
# # Initial search boxes (set manually).  These will be dynamically updated.
# red_box_UL = (307, 16)
# red_box_LR = (317, 26)
#
# green_box_UL = (308, 45)
# green_box_LR = (318, 55)
#
# # functions to update search box on fly
# # how much more intense is col than mean of other (RGB)?
# def av_color_intensity(img,x1,y1,x2,y2,col):
#     col_intensity = 0
#     for x in range(x1,x2+1):
#         for y in range(y1,y2+1):
#             col_intensity += img[y, x][col] - np.mean(img[y, x])
#     # return average
#     return col_intensity/( (x2-x1 + 1)*(y2-y1 + 1))
#
# def recentre_av_intensity(img,curr,col):
#     max_col_intensity = 0
#     max_col_intensity_where = (curr[0] + 5 - 2, curr[1] + 5 - 2)
#     sz = 4
#     for m in range(0,10):
#         for n in range(0,10):
#             col_intensity = av_color_intensity(img,curr[0]+m,curr[1]+n,curr[0]+m+sz,curr[1]+n+sz,col)
#             #print m,n,col_intensity
#             if col_intensity > max_col_intensity:
#                 max_col_intensity = col_intensity
#                 max_col_intensity_where = (m,n)
#     # get new upper left
#     new_ul = (max_col_intensity_where[0] - 5 + curr[0] + 2, max_col_intensity_where[1] - 5 + curr[1] + 2)
#     # ensure shift isn't too big
#     max_delta = 1 # alow max 1 px shift in each direction per frame
#     new_ul = (max(curr[0] - max_delta, new_ul[0]), max(curr[1] - max_delta, new_ul[1]))
#     new_ul = (min(curr[0] + max_delta, new_ul[0]), min(curr[1] + max_delta, new_ul[1]))
#     return [max_col_intensity, new_ul]
#
#
# for im_num in range(1,700):
#     img = cv2.imread('traffic_light.jpg',cv2.IMREAD_COLOR)
#     # get red intensity in existing red and green light area based on best 3x3 block (also returns location of best 3x3 block)
#     red_box = recentre_av_intensity(img,red_box_UL,2)
#     green_box = recentre_av_intensity(img,green_box_UL,1)
#     red_box_intensity = red_box[0]
#     green_box_intensity = green_box[0]
#     # if there is at least 33% more red or green then choose that light!
#     # update search area and add box
#     if red_box_intensity > 1.33*green_box_intensity:
#         red_box_UL = red_box[1]
#         red_box_LR = (red_box_UL[0]+10,red_box_UL[1]+10)
#         cv2.rectangle(img, red_box_UL, red_box_LR, (0,0,255), 2)
#         # cv2.circle(img, (int((red_box_UL[0] + red_box_LR[0])/2.0), int((red_box_UL[1] + red_box_LR[1])/2.0)), 5, (0,0,255), 2)
#         #cv2.rectangle(img, green_box_UL, green_box_LR, (255,255,255), 1) # search area for green!
#     elif green_box_intensity > 1.33*red_box_intensity:
#         green_box_UL = green_box[1]
#         green_box_LR = (green_box_UL[0]+10,green_box_UL[1]+10)
#         cv2.rectangle(img, green_box_UL, green_box_LR, (0,255,0), 2)
#
#         #cv2.rectangle(img, red_box_UL, red_box_LR, (255,255,255), 1) # search area for red
#     #else: # just put search boxes
#         #cv2.rectangle(img, green_box_UL, green_box_LR, (255,255,255), 1) # search area for green!
#         #cv2.rectangle(img, red_box_UL, red_box_LR, (255,255,255), 1) # search area for red
#     cv2.imshow('window_name', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




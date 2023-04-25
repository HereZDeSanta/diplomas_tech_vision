#ПЕРВЫЙ ТЕСТ(БЕЗ УСПЕХА)

# import cv2
# import numpy as np
#
# i = 6
# while (i<6):
#     frame = cv2.imread(str(i) ".jpg")
#     frame = cv2.resize(frame, (60, 120))
#     cv2.imshow(str(i), frame)
#
#     cutedFrame = frame[20:101, 8:52]
#     cv2.imshow("cutedFrame"+str(i), cutedFrame)
#
#     hsv = cv2.cvtColor(cutedFrame, cv2.COLOR_BGR2HSV)
#     v = hsv[:, :, 2]
#     cv2.imshow("v", + str(i), v)
#
#     red_sum = numpy.sum(v[0:27, 0:44])
#     yellow_sum = numpy.sum(v[28:54, 0:44])
#     green_sum = numpy.sum(v[55:81, 0:44])
#
#     cv2.rectangle(cutedFrame, (0,0), (44,27), (0,0,255), 2)
#     cv2.rectangle(cutedFrame, (0, 28), (44, 54), (0, 255, 255), 2)
#     cv2.rectangle(cutedFrame, (0, 55), (44, 81), (0, 255, 0), 2)
#
#     cv2.imshow("frameCopy"+str(i), cutedFrame)
#
#     print(str(red_sum) + " : " + str(yellow_sum) + " : " + str(green_sum))
#
#     if green_sum > yellow_sum and green_sum > red_sum:
#         print("green")
#     elif yellow_sum > green_sum and yellow_sum > red_sum:
#         print("yellow")
#     elif red_sum > green_sum and red_sum > yellow_sum:
#         print("red")
#     else:
#         print("red")
#
#
#     key = cv2.waitKey(1)
#     if key == ord("n"):
#         i = i+1





# # ТЕСТ РОИ
#
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture('3.mp4')
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     # frame = cv2.resize(frame, (1024, 1280))
#
#     roi = np.zeros_like(frame)
#     roi[100:100, 100:100] = frame[100:100, 100:100]
#
#     hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#
#     v = hsv[:, :, 2]
#
#     red_sum = np.sum(v[0:50, 0:50])
#     yellow_sum = np.sum(v[51:100, 0:50])
#     green_sum = np.sum(v[101:150, 0:50])
#
#     cv2.rectangle(frame, (200, 100), (400, 300), (0, 255, 0), 2)
#     cv2.rectangle(frame, (200, 100), (250, 150), (0, 0, 255), 2)
#     cv2.rectangle(frame, (200, 151), (250, 200), (0, 255, 255), 2)
#     cv2.rectangle(frame, (200, 201), (250, 250), (0, 255, 0), 2)
#
#     cv2.imshow('frame', frame)
#     cv2.imshow('roi', roi)
#
#     if green_sum > yellow_sum and green_sum > red_sum:
#         print("green")
#     elif yellow_sum > green_sum and yellow_sum > red_sum:
#         print("yellow")
#     else:
#         print("NOOOOOOOOO!")
#
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()





#
# ЧЕРЕЗ РОИ ПО ХСВ
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

    # frame = cv2.resize(frame, (720, 480))

    roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_sum = np.sum(red_mask)
    yellow_sum = np.sum(yellow_mask)
    green_sum = np.sum(green_mask)

    if red_sum > yellow_sum and red_sum > green_sum:
        color = 'red'
    elif yellow_sum > red_sum and yellow_sum > green_sum:
        color = 'yellow'
    elif green_sum > red_sum and green_sum > yellow_sum:
        color = 'green'
    else:
        color = 'none'

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)
    cv2.putText(frame, color, (roi_x, roi_y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

    cv2.imshow("Traffic Light Detection from Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# # ЧЕРЕЗ РОИ ПО ЛАБ
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture('3.mp4')
# percent = 100
#
# roi_x = 330
# roi_y = 430
# roi_width = 90
# roi_height = 110
#
# red_lower = np.array([0, 137, 249], dtype="uint8")
# red_upper = np.array([15, 255, 255], dtype="uint8")
# yellow_lower = np.array([17, 165, 130], dtype="uint8")
# yellow_upper = np.array([101, 255, 255], dtype="uint8")
# green_lower = np.array([40, 85, 180], dtype="uint8")
# green_upper = np.array([91, 255, 255], dtype="uint8")
#
# while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         break
#
#     #frame = cv2.resize(frame, (720, 480))
#
#     roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
#
#     hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#     lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
#     brightness = lab[:, :, 0]
#
#     brightness_mask = cv2.threshold(brightness, 50, 255, cv2.THRESH_BINARY)[1]
#
#     red_mask = cv2.inRange(hsv, red_lower, red_upper)
#     yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
#     green_mask = cv2.inRange(hsv, green_lower, green_upper)
#     mask_full = red_mask + yellow_mask + green_mask
#     # mask_full = cv2.bitwise_and(frame, frame, mask=mask_full)
#
#     red_weighted_sum = np.sum(cv2.bitwise_and(red_mask, brightness_mask) * brightness)
#     yellow_weighted_sum = np.sum(cv2.bitwise_and(yellow_mask, brightness_mask) * brightness)
#     green_weighted_sum = np.sum(cv2.bitwise_and(green_mask, brightness_mask) * brightness)
#     cv2.imshow("test red", red_mask)
#
#     # weighted_sums = {
#     #     "red": np.sum(cv2.bitwise_and(red_mask, brightness_mask) * brightness),
#     #     "yellow": np.sum(cv2.bitwise_and(yellow_mask, brightness_mask) * brightness),
#     #     "green": np.sum(cv2.bitwise_and(green_mask, brightness_mask) * brightness),
#     # }
#
#     if red_weighted_sum > yellow_weighted_sum and red_weighted_sum > green_weighted_sum:
#         color = 'red'
#     elif yellow_weighted_sum > red_weighted_sum and yellow_weighted_sum > green_weighted_sum:
#         color = 'yellow'
#     elif green_weighted_sum > red_weighted_sum and green_weighted_sum > yellow_weighted_sum:
#         color = 'green'
#     else:
#         color = 'none'
#
#     cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
#     cv2.putText(frame, color, (roi_x, roi_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#     cv2.imshow("Traffic Light Detection from Video", frame)
#     # cv2.imshow("LAB", weighted_sums)
#
#     if cv2.waitKey(1) and 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



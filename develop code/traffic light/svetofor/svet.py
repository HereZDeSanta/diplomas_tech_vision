import numpy as np
import cv2 
import math as m

cap = cv2.VideoCapture(0)
     
while(cap.isOpened()):
    ret, frame = cap.read()    
    frame=cv2.imread('3.mp4',cv2.IMREAD_COLOR)
    cv2.imshow("video_frame", frame)   
    kernel=np.ones((3,3),np.uint8)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 110], dtype = "uint8")
    upper_red = np.array([15, 255, 255], dtype = "uint8")
    lower_violet = np.array([165, 150, 110], dtype = "uint8")
    upper_violet = np.array([180, 255, 255], dtype = "uint8")
    red_mask_orange = cv2.inRange(frame_hsv, lower_red, upper_red)      
    red_mask_violet = cv2.inRange(frame_hsv, lower_violet, upper_violet)  
    red_mask_full = red_mask_orange + red_mask_violet 
    # contours, hierarcy = cv2.findContours(red_mask_full, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("mask_hsv", red_mask_full)
    # pic_erode = cv2.erode(red_mask_full, kernel, iterations=1)
    pic3 = cv2.GaussianBlur(red_mask_full, (9, 9), 4)
    edges = cv2.Canny(pic3, 130,85, 7)
    pic1 = cv2.dilate(edges, kernel, iterations=4)
    cv2.imshow("mask_dilation", pic1)
    # a = frame.shape[0]
    # b=frame.shape[1]
    # print([a,b])
    contours, hierarcy = cv2.findContours(pic1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        (x,y),radius = cv2.minEnclosingCircle(contours[i])
        # print(x,y)
        if x<600 and y<500:
            if cv2.contourArea(contours[i])>100:
                con = cv2.drawContours(frame,contours,i,[0,255,0], 3)
                # cv2.putText(frame,'STOP',(50,90),cv2.FONT_HERSHEY_COMPLEX,3,[0,0,255],5)
                stop=1
                cv2.imshow('contours2', con)
            else:
                stop=0
                # cv2.putText(frame,'GO',(50,90),cv2.FONT_HERSHEY_COMPLEX,3,[0,0,255],5)
                cv2.imshow('contours2', frame)
            print(stop)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# # print(contours)
# # for i in range(len(contours)):
# #      # con = cv2.drawContours(frame,contours,i,[0,255,0], 3)
# #      (x,y),radius = cv2.minEnclosingCircle(contours[i])
# #      print('x',x)
# #      print('y',y)
# #      circle_square=m.pi*radius*radius
# #      # cv2.circle(frame,(int(x),int(y)),int(radius),[255,255,0],3)
# #      area = cv2.contourArea(contours[i])
# #      if radius >=20:
# #         if abs(area-circle_square)<=300 and abs(area-circle_square)>=5:
# #             maybe_red=contours[i]
# #             con2 = cv2.drawContours(frame,contours,i,[255,0,0], 3)
# #             cv2.imshow('contours2', con2)

# # cv2.imshow('contours',red_mask_full)
# # cv2.waitKey(0)
# # # cv2.imshow("rrr", red_mask_full)
# # # a = red_mask_full.shape[0]
# # # circles = cv2.HoughCircles(red_mask_full, cv2.HOUGH_GRADIENT, 1, minDist = a//40, param1 = 10, param2 = 5, minRadius = a//1000, maxRadius = a//2) 
# # # print(circles[0])
# # # for circle in circles[0]:
# # #     center = int(circle[0]), int(circle[1])
# # #     radius = int(circle[2])
# # # cv2.circle(frame, center, radius, (0, 0, 255), 3) 
# # # cv2.imshow("rrr", frame)
# # cv2.waitKey(0)
# # # cv2.destroyAllWindows()

# Раздельные окна (хуевасто отрабатывает)
# import cv2
#
# # Create a video capture object for the ELP960 stereo camera
# cap = cv2.VideoCapture(1)
#
# # Set the width and height of the camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# if not cap.isOpened():
#     print("Error opening video")
#
# while (cap.isOpened()):
#     # Read the next stereo image pair
#     ret, frame = cap.read()
#     img_right = frame[:, 640:, :]
#     img_left = frame[:, 640:, :]
#
#     # Display the left and right images
#     cv2.imshow('Left', img_left)
#     cv2.imshow('Right', img_right)
#
#     # Check for keyboard input to exit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()



#Конкатенация в одно окно (отрабатывает неплохо)
# import cv2
# import numpy as np
#
# # Create a video capture object for the ELP960 stereo camera
# cap = cv2.VideoCapture(1)
#
# # Set the width and height of the camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# if not cap.isOpened():
#      print("Error opening video")
#
# while(cap.isOpened()):
#      # Read the next stereo image pair
#      ret, frame = cap.read()
#      img_right = frame[:, 640:, :]
#      img_left = frame[:, :640, :]
#
#      # Concatenate the left and right images horizontally
#      stereo_img = np.hstack((img_left, img_right))
#
#      # Resize the output window
#      stereo_img = cv2.resize(stereo_img, (640, 360))
#
#      # Display the stereo image
#      cv2.imshow('Stereo', stereo_img)
#
#      # Check for keyboard input to exit the loop
#      if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
#
# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


# Текст распознавания сигналов с камеры elk960
import cv2
import numpy as np

# Create a video capture object for the ELP960 stereo camera
cap = cv2.VideoCapture(1)

# Set the width and height of the camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
     print("Error opening video")

# Define the color range for detecting red traffic signals
lower_red = np.array([0, 137, 249], dtype="uint8")
upper_red = np.array([15, 255, 255], dtype="uint8")
lower_yellow = np.array([17, 165, 130], dtype="uint8")
upper_yellow = np.array([101, 255, 255], dtype="uint8")
lower_green = np.array([40, 85, 180], dtype="uint8")
upper_green = np.array([91, 255, 255], dtype="uint8")

while(cap.isOpened()):
     # Read the next stereo image pair
     ret,frame = cap.read()

     # Split the stereo image into left and right frames
     img_left = frame[:, 0:640, :]
     img_right = frame[:, 640:, :]

     # Resize the left and right frames to half their original size
     img_left = cv2.resize(img_left, (0, 0), fx=0.5, fy=0.5)
     img_right = cv2.resize(img_right, (0, 0), fx=0.5, fy=0.5)

     # Concatenate the left and right frames into a single image
     combined = np.concatenate((img_left, img_right), axis=1)

     # Convert the combined image to HSV color space
     hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)

     # Create a binary mask for the red color range
     mask_red = cv2.inRange(hsv, lower_red, upper_red)
     mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
     mask_green = cv2.inRange(hsv, lower_green, upper_green)

     mask_full = mask_red + mask_yellow + mask_green
     mask_full = cv2.bitwise_and(frame, combined, mask=mask_full)
     cv2.imshow("mask_hsv", mask_full)

     # Find contours in the binary mask
     contours, hierarchy = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     # Draw a bounding box around each red traffic signal
     for cnt in contours:
         area = cv2.contourArea(cnt)
         if area > 100:
             x, y, w, h = cv2.boundingRect(cnt)
             cv2.rectangle(combined, (x, y), (x + w, y + h), (0, 0, 255), 2)

     # Display the concatenated image with bounding boxes around red traffic signals
     cv2.imshow('Traffic Signals', combined)

     # Check for keyboard input to exit the loop
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

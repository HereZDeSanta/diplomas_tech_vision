import cv2
import numpy as np

def canny(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    return cv2.Canny(blur, 50, 150)

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):

    left_fit = []
    right_fit = []

    while lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        # i = 1
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 8)
            # if i == 1:
            #     cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 8)
                # i += 1
            # else:
            #     cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 8)
                # pts = np.array([[[pl[0], pl[1]], [pl[2], pl[3]], [x2, y2], [x1, y1]]], dtype=np.init32)
                # cv2.fillPoly(line_image, pts, (202, 255, 191), lineType=8, shift=0, offset=None)
    return line_image


def ROI(image):
    height = image.shape[0]
    polygons = np.array([(600, height//2), (1000, height//2), (800, 700), (0, 1000)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
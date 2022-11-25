import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import collections.abc


def average_lines(lines):
    left_line_avg = []
    right_line_avg = []
    try:
        for line in lines:
            line = line.reshape(4)
            (x1, y1, x2, y2) = line
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            length = math.sqrt(pow(x2-x1, 2)+pow(y2-y1, 2))
            slope = parameters[0]
            intercept = parameters[1]
            if(slope > 0):
                right_line_avg.append((slope, intercept))
            else:
                left_line_avg.append((slope, intercept))
        left_line_avg = np.average(left_line_avg, axis=0)
        right_line_avg = np.average(right_line_avg, axis=0)
        return np.array([left_line_avg, right_line_avg])
    except:
        return np.array([(0, 0), (0, 0)])


def getCoordinates(image, parameters):
    y1 = image.shape[0]
    y2 = int(y1*(3/5))

    if(type(parameters) is np.ndarray):
        x1 = int((y1-parameters[1])/parameters[0])
        x2 = int((y2-parameters[1])/parameters[0])
        return np.array([[x1, y1, x2, y2]])
    else:
        return np.array([[0, 0, 0, 0]])


def draw_lines(image, lines):
    base = np.zeros_like(image)

    for line in lines:
        x1 = line[0, 0]
        y1 = line[0, 1]
        x2 = line[0, 2]
        y2 = line[0, 3]
        try:
            cv2.line(base, (int(x1), int(y1)),
                     (int(x2), int(y2)), (255, 0, 0), 5)
        except:
            print("error")

    return base


def canny_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def roi(image):
    height = image.shape[0]
    mask = np.zeros_like(image)
    polygons = np.array(
        [[(200, height), (1100, height), (550, 250)]])
    cv2.fillPoly(mask, polygons, 255)
    masked_img = cv2.bitwise_and(mask, image)
    return masked_img


tr = {}
index = 1


vid = cv2.VideoCapture("./videos/test2.mp4")

while(vid.isOpened()):

    ret, frame = vid.read()

    if ret == True:
        np_arr = np.copy(frame)

        canny = canny_detection(np_arr)

        cropped_img = roi(canny)

        lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100,
                                np.array([]), minLineLength=50, maxLineGap=10)
        avg_lines = average_lines(lines)
        leftCoordinates = getCoordinates(frame, avg_lines[0])
        rightCoordinates = getCoordinates(frame, avg_lines[1])
        lanes = draw_lines(np_arr, np.array(
            [leftCoordinates, rightCoordinates]))
        merged_img = cv2.addWeighted(np_arr, 0.8, lanes, 1, 1)
        cv2.imshow('Result', merged_img)
        if(cv2.waitKey(1) == ord('q')):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows()

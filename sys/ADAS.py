import yolo
import laneNet
import cv2
vid = cv2.VideoCapture("./videos/solidWhiteRight.mp4")

while(vid.isOpened()):
    ret, frame = vid.read()
    if(ret == True):
        lane_image = laneNet.laneDetector(frame)
        object_image = yolo.YoloObjectDetection(lane_image)
        cv2.imshow('result', object_image)
        if(cv2.waitKey(1) == ord('q')):
            break
    else:
        break

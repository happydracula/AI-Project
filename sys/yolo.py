import cv2
import numpy as np
import matplotlib.pyplot as plt
net = cv2.dnn.readNet("yolo\yolov4-tiny.weights", "yolo\yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(500, 500), scale=1/255)


def load_classes(filename):
    file = open(filename, 'r')
    classes = (file.readlines())
    res = []
    for cl in classes:
        res.append(cl.replace("\n", ""))
    return res


res = load_classes("yolo\classes.txt")


def non_max_suppression(classIds, confidences, boxes):
    b = []
    cids = []
    conf = []
    for i in range(len(confidences)):
        if(confidences[i] > 0.6):
            b.append(boxes[i])
            cids.append(classIds[i])
            conf.append(confidences[i])
    boxes = np.array(b)
    confidences = np.array(conf)
    classIds = np.array(cids)
    resIDS = []
    resConfidences = []
    resBoxes = []
    while(len(b) != 0):
        max_index = np.argmax(conf)
        box = b.pop(max_index)
        confidence = conf.pop(max_index)
        cid = cids.pop(max_index)
        resIDS.append(cid)
        resConfidences.append(confidence)
        resBoxes.append(box)
        cids = [cids[i] for i in range(len(cids)) if iou(box, b[i]) < 0.5]
        conf = [conf[i] for i in range(len(conf)) if iou(box, b[i]) < 0.5]
        b = [i for i in b if iou(box, i) < 0.5]

    return (np.array(resIDS), np.array(resConfidences), np.array(resBoxes))


def iou(box1, box2):
    x11, y11, br1, h1 = box1
    x21, y21, br2, h2 = box2
    b1 = {}
    b2 = {}
    if(x11 <= x21):
        b1['x1'] = x11
        b1['y1'] = y11
        b1['x2'] = x11+br1
        b1['y2'] = y11+h1
        b2['x1'] = x21
        b2['y1'] = y21
        b2['x2'] = br2+x21
        b2['y2'] = h2+y21
    else:
        b2['x1'] = x11
        b2['y1'] = y11
        b2['x2'] = br1+x11
        b2['y2'] = h1+y11
        b1['x1'] = x21
        b1['y1'] = y21
        b1['x2'] = br2+x21
        b1['y2'] = h2+y21
    u = (b1['x2']-b1['x1'])*(b1['y2']-b1['y1']) + \
        (b2['x2']-b2['x1'])*(b2['y2']-b2['y1'])
    i = 0
    if(b2['x1'] >= b1['x2'] or b2['x1'] < b1['x1'] and (b2['y1'] >= b1['y2'] or b2['y2'] <= b1['y1'])):
        i = 0
    else:
        ib = {}
        ib['x1'] = b2['x1']
        ib['y1'] = b1['y1'] if b1['y1'] >= b2['y1'] else b2['y2']
        ib['x2'] = b1['x2'] if b1['x2'] <= b2['x2'] else b2['x2']
        ib['y2'] = b1['y2'] if b1['y2'] <= b2['y2'] else b2['y2']
        i = (ib['x2']-ib['x1'])*(ib['y2']-ib['y1'])
    u = u-i

    return i/u


def drawBoxes(img, classIds, confidences, boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        (x1, y1, b, h) = box
        x2 = x1+b
        y2 = y1+h
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.putText(img, (res[classIds[i]])+str(" ")+str(confidences[i]), (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img


# vid = cv2.VideoCapture("./videos/car_drive.mp4")

# while(vid.isOpened()):
#     _, frame = vid.read()

#     frame = cv2.resize(frame, (500, 500))
#     img = np.copy(frame)
#     (classIds, confidences, boxes) = model.detect(img)
#     (classIds, confidences, boxes) = non_max_suppression(
#         classIds, confidences, boxes)
#     for i in range(len(boxes)):
#         for j in range(len(boxes)):
#             if(i != j):
#                 print(iou(boxes[i], boxes[j]))
#     result = drawBoxes(img, classIds, confidences, boxes)
#     cv2.imshow('result', result)
#     cv2.waitKey(1)
frame = cv2.imread('./images/4.jpg')
frame = cv2.resize(frame, (500, 500))
img = np.copy(frame)
(classIds, confidences, boxes) = model.detect(img)
(classIds, confidences, boxes) = non_max_suppression(
    classIds, confidences, boxes)
for i in range(len(boxes)):
    for j in range(len(boxes)):
        if(i != j):
            print(iou(boxes[i], boxes[j]))
result = drawBoxes(img, classIds, confidences, boxes)
cv2.imshow('result', result)
cv2.waitKey(0)

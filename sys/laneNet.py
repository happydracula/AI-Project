import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
model = keras.models.load_model(
    filepath="./LaneNet/full_CNN_model.h5")
image = cv2.imread("./images/test_image.jpg")
original_shape = image.shape
original_image = np.copy(image)
print(image.shape)


def resize_image(image):
    # print(image.shape)
    image = cv2.resize(image, (160, 80))
    # print(image.shape)
    small_img = cv2.resize(image, (160, 80))
    # print(plt.imshow(image))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]
    return small_img, image

    # print(small_img.shape)


def laneDetector(image):
    original_image = np.array(image)
    original_image_shape = original_image.shape
    small_img, image = resize_image(image)
    res = model.predict(small_img)
    lane = res[0]*255

    lane = cv2.resize(
        lane, (original_image_shape[1], original_image_shape[0]))
    print(lane.shape)
    blanks = np.zeros((lane.shape[0], lane.shape[1], 3))
    blanks[:, :, 1] = blanks[:, :, 1]+lane[:, :]

    lane_image = blanks.astype('uint8')

    result = cv2.addWeighted(original_image, 1, lane_image, 1, 0)
    return result


# vid = cv2.VideoCapture("./videos/solidWhiteRight.mp4")
# while(vid.isOpened()):
#     ret, image = vid.read()

#     if ret == True:

#         result = laneDetector(image)
#         cv2.imshow('result', result)

#         # print(plt.imshow(result))
#         if(cv2.waitKey(1) == ord('q')):
#             break
#     else:
#         break


# vid.release()
# cv2.destroyAllWindows()

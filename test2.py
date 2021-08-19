import tensorflow as tf
import numpy as np
from keras.datasets import mnist
import cv2

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0])


x_train= x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test= x_test.reshape((x_test.shape[0], 28, 28, 1))

print(x_train[0])

image = x_train[0].copy()

def BnW(image):
    originalImage = image.copy()
    grayImage = originalImage.copy() #cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    #blackAndWhiteImage = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_BGR2RGB)
    blackAndWhiteImage = blackAndWhiteImage.reshape(28, 28, 1)

    return blackAndWhiteImage


changed = BnW(image)

print(changed)
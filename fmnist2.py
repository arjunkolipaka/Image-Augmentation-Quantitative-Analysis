import cv2
import time
import statistics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import tensorflow.keras.utils
import imgaug.augmenters as iaa
import tensorflow_addons as tfa
from keras.datasets import fashion_mnist
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Importing and loading the MNIST dataset from keras.datasets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train[0:6000]
y_train = y_train[0:6000]
x_test = x_test[0:1000]
y_test = y_test[0:1000]



x_train= x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test= x_test.reshape((x_test.shape[0], 28, 28, 1))
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


t1 = time.time() #To measure time.

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-4),
              metrics=['accuracy'])


#Average Blur
def avg_blur(image):
    image = tfa.image.mean_filter2d(
        image, 
        (3, 3)
        )

    return image


#Gaussian Blur
def g_blur(image):
    image = tfa.image.gaussian_filter2d(
        image,
        (3, 3),
        sigma = 5.0     #Note: sigma = 0 is pitch black. Skip it.
        )

    return image


#Sharpen from imgaug
def Sharp(image):
    aug = iaa.Sharpen(alpha=0.1, lightness=1.0)
    image = aug(images = image)

    return image


#Salt and Pepper Noise
def SnPnoise(image):
    aug = iaa.SaltAndPepper(0.1)
    image = aug(images = image)

    return image


#Similar to Random Erasing
def Coarse_Dropout(image):
    aug = iaa.CoarseDropout(0.1)
    image = aug(images = image)
    
    return image


#Black-and-White Image
def BnW(image):
    originalImage = image.copy()
    grayImage = originalImage.copy() #cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY) for Coloured Images
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY)
    #blackAndWhiteImage = cv2.cvtColor(blackAndWhiteImage, cv2.COLOR_BGR2RGB) for multiple channels (Coloured Images)
    blackAndWhiteImage = blackAndWhiteImage.reshape(28, 28, 1)

    return blackAndWhiteImage


#HERE YOU CAN TO CHANGE/ADD/REMOVE PARAMETERS 
def evaluate_model(x_train, y_train, x_test, y_test, model, param1, n_folds=5):
    scores, histories = list(), list()
    model = model

    #Image Augmentation here
    datagen = ImageDataGenerator(rescale=1./255, preprocessing_function = Coarse_Dropout)

    #Kfold crossvalidation for improving accuracy of the model
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(x_train):
        history = model.fit(datagen.flow(x_train[train_ix], y_train[train_ix], batch_size=20), validation_data=datagen.flow(x_train[test_ix], y_train[test_ix], batch_size=20), epochs=5)
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        acc = acc * 100
        scores.append(acc)
        histories.append(history) 
    accuracy = statistics.mean(scores)
    print(accuracy)
    print(scores)
    return scores, histories, accuracy


#List to keep track of accuracies of different values of each parameter
acc = list()

#Use either np.arange() function or a list of values
#for a single value use 1 value in a list like so : [a]
for i in [1]:
  _,_,a = evaluate_model(x_train, y_train, x_test, y_test, model, i,  n_folds=3)
  acc.append([i,a])

print(acc)

t2 = time.time()
print(t2-t1)
#Time taken for the whole execution cycle

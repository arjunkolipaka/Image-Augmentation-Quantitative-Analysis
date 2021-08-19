#gaussian blur
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import skimage
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_io as tfio
import imgaug.augmenters as iaa
#import gi
#gi.require_version('Gtk', '2.0')

image = io.imread("/home/arjun/Pictures/redpanda.png", as_gray=False)
image = tf.convert_to_tensor(image, dtype=int8)


def Saturation(image):
	saturation_factor = 0.5
	image = tf.image.grayscale_to_rgb(image)
	image = tf.image.adjust_saturation(image, saturation_factor)
	image = tf.image.rgb_to_grayscale(image)
	
	return image


t = tf.shape(image)
print(t)
image = Saturation(image)


#Changed Image
arr = image.numpy()
#arr = image
arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr_)
plt.show()




#image1 = image
#image2 = image
"""image1 = avg_blur(image)
#Changed Image1
arr = image1.numpy()
arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr_)
plt.show()"""

"""image2 = laplacian_sharpening(image2)
#Changed Image2
arr = image1.numpy()
arr_ = np.squeeze(arr) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr_)
plt.show()"""

#print(image)

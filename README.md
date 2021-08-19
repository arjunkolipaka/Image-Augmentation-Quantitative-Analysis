# Image-Augmentation-Quantitative-Analysis

An [older version of the] code part of the work we did on Image Augmentation.

This repo consists of an earlier version of the code that was written for experimentation part of our work in Quantitative Analysis - Image Augmentation. To reproduce the same, use:
```
$ python3 mnist.py
```

## Image Data Generator

The ImageDataGenerator api from the Keras allows us to perform/generate data augmentation to the batches of image data in real-time. We have 18 different data augmentations ready to use, built into the ImageDataGenerator class.
```
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    dtype=None,
)
```
Visit the Keras documentation for more info: [🔗](https://keras.io/api/preprocessing/image/#imagedatagenerator-class)

## Writing Your Own Augmentation Function

To define a custom augmentation, as we did for sharpening, blurring, colour shift etc. you can write you own custom augmentation in a function then use it by passing the fuction into the ```preprocessing_function=None``` parameter in ImageDataGenerator.

Below is an example where we implement Averaging blur:
```
import tensorflow_addons as tfa
def avg_blur(image):
    image = tfa.image.mean_filter2d(
        image, 
        filter_shape = (3, 3)
        )

    return image
```
The above function returns a tf.Tensor which is then passed into the model to train on the data generated. 

## Using Multiple Custom Augmentations

We can apply multiple augmentations on an image by stacking the augmentations in the order you'd wish to be used in and put it in a function. For example,

```
def multi_aug():
    image = avg_blur(image)
    image = sharpening(image)
    image = noise(image)
    
    return image
```

## Dependencies

* Tensorflow == 2.6.0
* Keras == 2.6.0
* OpenCV == 4.5.3
* numpy
* Matplotlib
* TensorFlow Addons
* imgaug == 0.4.0 [🔗](https://github.com/aleju/imgaug)

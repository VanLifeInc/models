import glob
import os

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from keras_applications import mobilenet_v2

def load_images(image_types=None,
                directory=None,
                images_per_type=None,
                image_size=224,
                process=False,
                model=mobilenet_v2):
    """
    Load images and classes from a directory with the option to process
    the images to be used by a model.

    :param list[str] image_types: classes of images to load
    :param str directory: location of folders for each image class
    :param int images_per_type: number of images to load for each class
    :param int image_size: length and width of the loaded images
    :param bool process: option to process the images so that they can
        be used to train and test the model
    :param keras.application model: model for which to process the
        images
    :return list[list]: loaded images 
    :return list[str]: image classes
    """

    images_numpy = []
    images_class = []

    for image_type in image_types:
        images_path = os.path.join(directory, image_type, '*.jpg')
        for i, filename in enumerate(glob.glob(images_path)):
            try:
                if i == images_per_type:
                    break
                loaded_image = load_img(filename, target_size=(image_size, image_size))
                images_numpy.append(img_to_array(loaded_image))
                images_class.append(image_type)
            except Exception as e:
                print('TypeError: {}'.format(e))

    if process:
        image_batch = np.expand_dims(images_numpy, axis=0)
        images_processed = model.preprocess_input(image_batch.copy())
        images_class_processed = process_images_class(images_class)

        return images_processed[0], images_class_processed

    else:
        return images_numpy, images_class


def process_images_class(images_class):
    """
    Convert images_class to a binary class matrix. This creates the
    target values for the model.

    :param list[str] images_class: the class of each image
    :return list[list[int]]: the labels for each image
    """

    images_class_to_int_map = {image: i for (i, image) in enumerate(sorted(set(images_class)))}
    print('Mapping:\n{}'.format(images_class_to_int_map))
    images_class_int = [images_class_to_int_map[i] for i in images_class]

    return to_categorical(images_class_int, num_classes=len(set(images_class)))

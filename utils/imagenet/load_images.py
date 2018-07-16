import glob
import os

import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

from utils.imagenet import scraper


def get_imagenet_data(download=False,
                      download_info=None,
                      directory=None,
                      n_images=None,
                      image_size=224,
                      process=False,
                      model=None):

    if download:

        for imagenet_id, image_type in download_info.items():
            images_directory = os.path.join(directory, image_type)
            scraper.main(imagenet_id, n_images, images_directory)

        print('Images have finished downloading')

    # Load images
    images_numpy = []
    images_class = []

    for image_type in download_info.values():
        images_path = os.path.join(directory, image_type, '*.jpg')
        for filename in glob.glob(images_path):
            try:
                loaded_image = load_img(filename, target_size=(image_size, image_size))
                images_numpy.append(img_to_array(loaded_image))
                images_class.append(image_type)
            except:
                continue

    if process:
        image_batch = np.expand_dims(images_numpy, axis=0)
        images_processed = model.preprocess_input(image_batch.copy())
        images_class_processed = process_images_class(images_class)

        return images_processed[0], images_class_processed

    else:
        images_numpy, images_class


def process_images_class(images_class):

    images_class_to_int_map = {image: i for (i, image) in enumerate(set(images_class))}
    print('Mapping:\n{}'.format(images_class_to_int_map))
    images_class_int = [images_class_to_int_map[i] for i in images_class]

    return to_categorical(images_class_int, num_classes=len(set(images_class)))

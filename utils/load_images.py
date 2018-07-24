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

    images_class_to_int_map = {image: i for (i, image) in enumerate(sorted(set(images_class)))}
    images_class_to_int_map = {'bed': 0, 'lamp': 1, 'couch': 2, 'table': 3, 'chair': 4}
    print('Mapping:\n{}'.format(images_class_to_int_map))
    images_class_int = [images_class_to_int_map[i] for i in images_class]

    return to_categorical(images_class_int, num_classes=len(set(images_class)))

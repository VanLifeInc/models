import os

from utils.imagenet import scraper
from utils.load_images import load_images


def get_imagenet_data(download=False,
                      download_info=None,
                      directory=None,
                      images_per_type=None,
                      image_size=224,
                      process=False,
                      model=None):
    """
    Optionally download a set number of images from ImageNet, then load
    the images and classes.

    :param bool download: choice to download the images
    :download_info dict(str: str): WordNet ID and associated image type
        for each class of image to download
    :param str directory: where to create a directory for each image
        type
    :param int images_per_type: number of images to download for each
        image type
    :param int image_size: length and width of the loaded images
    :param bool process: option to process the images so that they can
        be used to train and test the model
    :param keras.application model: model for which to process the
        images
    :return list[list]: images from ImageNet
    :return list[str]: image classes
    """

    if download:

        for imagenet_id, image_type in download_info.items():
            images_directory = os.path.join(directory, image_type)
            scraper.main(imagenet_id, images_per_type, images_directory)

        print('Images have finished downloading')

    images, classes = load_images(image_types=download_info.values(),
                                  directory=directory,
                                  images_per_type=images_per_type,
                                  image_size=image_size,
                                  process=process,
                                  model=model)

    return images, classes

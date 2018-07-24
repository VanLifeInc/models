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

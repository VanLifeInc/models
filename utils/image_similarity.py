import numpy as np
from scipy.spatial import distance


def get_closest_images(images, image_index_to_measure, num_results=5):
    """
    Calculate the manhattan distance between the image of
    image_index_to_measure and all other images. Return the indicies
    of the closest images and their distances.

    :param list(list(int)) images: images to measure distance to
    :param int image_index_to_measure: index of image with which to
        measure distances
    :param int num_results: number of distances and indicies to return
    :return list(float) distances_closest: distances of closest images
    :return list(int) indicies_closest: indicies of closest images
    """

    distances = [distance.cityblock(images[image_index_to_measure], image)
                 for image in images]
    indicies_closest = sorted(
        range(len(distances)), key=lambda k: distances[k]
    )[1:1+num_results]

    distances_closest = sorted(distances)[1:1+num_results]

    return distances_closest, indicies_closest


def get_concatenated_images(images, image_indicies_to_concatenate):
    """
    Create a row of concatenated images.

    :param list(list(float)) images: a list of images
    :param list(int) image_indicies_to_concatenate: indicies of images
        to concatenate
    :return list(list(float)): the desired images concatenated
    """

    concat_image = np.concatenate(
        [np.uint8(images[index]) for index in image_indicies_to_concatenate],
        axis=1
    )
    return concat_image

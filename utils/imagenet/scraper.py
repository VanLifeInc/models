"""
Download a specified number of images from the ImageNet database.
"""

import defusedxml.ElementTree
import frogress
import grequests
import os
import re
import requests
import sys
import urllib.parse

MAIN_PAGE_URL = 'http://image-net.org/synset?wnid={}'
SYNSET_INDEX_URL = 'http://image-net.org/python/gp.py/ImagesXML?type=synsetgood&synsetid={}&start=0&n={}'
IMAGENET_THUMB_URL = 'http://image-net.org/nodes/{}/{}/{}/{}.thumb'


def verify_wnid(wnid):
    """
    Verify that the WordNet ID (wnid) is valid.

    :param str wnid: WordNet ID to be validated
    :return str: validated WordNet ID
    """
    match = re.search('^n[0-9]{8}$', wnid.lower())
    if match:
        return wnid
    else:
        print('{} is not a valid synset id; example: n00007846'.format(wnid))


def get_wnid_synsets_id(wnid):
    """
    Get the synsets ID for a WordNet ID.

    :param str wnid: the WordNet ID for the desired synsets ID
    :return str: the synsets ID for the searched WordNet ID
    """
    r = requests.get(MAIN_PAGE_URL.format(wnid))
    target_id_pattern = '^target_id = \'([0-9]+)\';$'
    match = re.search(target_id_pattern, r.content.decode('utf-8'), re.M)
    if not match:
        raise Exception('Could not find target id in: {}'.format(r.url))
    return int(match.group(1))


def get_number_of_images(synsets_id):
    """
    Get the number of images for a synsets ID.

    :param str synsets_id: the synsets ID to look up
    :return int: number of images in ImageNet for the synsets ID
    """
    r = requests.get(SYNSET_INDEX_URL.format(synsets_id, 0))
    return int(defusedxml.ElementTree.fromstring(r.content)
                                     .find('imageset')
                                     .attrib['total'])


def get_image_data(synsets_id, n_images):
    """
    Get the request data for a specificed number of images for a
    synsets ID.

    :param str synsets_id: the synsets ID for which to get images
    :param int n_images: the number of images to get
    :return _elementtree._element_iterator: contains the data to
                                            download the images
    """
    r = requests.get(SYNSET_INDEX_URL.format(synsets_id, n_images))
    return (defusedxml.ElementTree.fromstring(r.content)
                                  .find('imageset')
                                  .iter('image'))


def make_thumb_url(image_data):
    """
    Fill in the required fields for an images thumbnail url.

    :param xml.etree.ElementTree.Element image_data: data relating to
                                                     an image
    :return str: the compete url for an image's thumbnail
    """
    return IMAGENET_THUMB_URL.format(image_data.attrib['node'],
                                     image_data.attrib['synsetoffset'],
                                     image_data.attrib['prefix'][:2],
                                     image_data.attrib['prefix'])


def download_iamges(image_data, n_images, output_dir):
    """
    Download a specified number of images to out_dir.

    :param _elementtree._element_iterator image_data: information to
                                                      download images
    :param int n_images: number of images to download
    :param str output_dir: directory to store the images
    """

    urls = (make_thumb_url(image) for image in image_data)
    reqs = (grequests.get(url) for url in urls)
    responses = grequests.imap(reqs)

    responses = frogress.bar(responses, steps=n_images)
    print('\nDownloading {} images'.format(n_images))

    os.makedirs(output_dir, exist_ok=True)

    for r in responses:
        try:
            url = urllib.parse.urlparse(r.url)
            filename, _ = os.path.splitext(os.path.basename(url.path))
            output_file_path = os.path.join(output_dir, filename + '.jpg')
            with open(output_file_path, 'wb') as output_file:
                output_file.write(r.content)
        finally:
            r.close()


def main(wnid, image_limit, output_dir):
    """
    Download a specified number of images from the ImageNet database.

    :param str wnid: WordNet ID of images to download
    :param int image_limit: number of images to download
    :param str output_dir: directory to download images to
    """

    verify_wnid(wnid)
    target_id = get_wnid_synsets_id(wnid)
    n_images = get_number_of_images(target_id)
    if image_limit < n_images:
        n_images = image_limit
    images = get_image_data(target_id, n_images)
    download_iamges(images, n_images, output_dir)


if __name__ == '__main__':

    wnid = sys.argv[1]
    image_limit = sys.argv[2]
    output_dir = sys.argv[3]

    main(wnid, image_limit, output_dir)

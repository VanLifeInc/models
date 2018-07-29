import os
from codecs import open

from setuptools import find_packages, setup


HERE = os.path.abspath(os.path.dirname(__file__))
PATH_VERSION = os.path.join(HERE, '__version__.py')

about = {}
with open(PATH_VERSION, 'r', 'utf-8') as f:
    exec(f.read(), about)

setup(
    name='models',
    version='0.0.2',
    long_description='Van Life Inc.\'s modelling repo',
    install_requires=[
        'defusedxml==0.5.0',
        'flake8==3.5.0',
        'frogress==0.9.1',
        'grequests==0.3.0',
        'Keras==2.2.0',
        'Keras-Applications==1.0.2',
        'Keras-Preprocessing==1.0.1',
        'matplotlib==2.2.2',
        'numpy==1.14.0',
        'pandas==0.23.0',
        'requests==2.18.4',
        'scikit-learn==0.18.1',
    ],
)
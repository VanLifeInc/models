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
    version='0.0.1',
    long_description='Van Life Inc.\'s modelling repo',
    install_requires=[
        'numpy==1.14.0',
        'pandas==0.23.0',
    ],
)
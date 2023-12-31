from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.01'
DESCRIPTION = 'Dataset size reduction using KNN Sampling algorithm'
LONG_DESCRIPTION = 'An implementation of KNN based Sampling algorithm for faster and better visualization of large datasets by selecting representatives without loss of general pattern/relation'

# Setting up
setup(
    name="KNearestNeighborSampling",
    version=VERSION,
    author="Soham S. Nimale",
    url="https://github.com/snimale/KNNSampler",
    author_email="soham.sachin.nimale@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'sklearn'],
    keywords=['python', 'K-NN', 'Sampling', 'Size reduction', 'Optimization', 'knn sampler'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ]
)
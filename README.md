# Image Classification
This repo implements an image classification model using different backbones (ResNet, DenseNet, MobileNet, ...)

## Installation
To install all dependencies and set up the Python environment, we recommend using [poetry](https://python-poetry.org/docs/).

For more information about poetry, visit the official documentation.

To install the dependencies, run the following command:

    poetry install

To enter the environment in the shell, run:

    poetry shell

To verify that everything is working, start a Python interpreter in the command line and run:

    import image_classifier

Alternatively, you can use any package manager of your choice. The main dependencies can be found in the [toml file](pyproject.toml) file.

## Running the Script
The main script that runs the algorithm is located here. To run it, execute the following command:


    python scripts/main.py

Please make sure that you have all the dependencies installed before running the script.

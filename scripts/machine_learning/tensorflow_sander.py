#!/usr/bin/env python3

"""
Script to segment a given image using a given classifier
"""

__author__ = "Skippybal & Reindert1"
__version__ = "0.1"

import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import cv2 as cv
import gc
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def model_to_tif(model_file, save_loc, dataset_loc, palette, original_image_loc):
    loaded_model = keras.models.load_model(model_file)
    loaded_model.summary()
    x_data = np.load(dataset_loc, mmap_mode="r")
    x_data = x_data[:, :-1]

    predictions = loaded_model.predict(x_data)

    image_array = cv.imread(original_image_loc, cv.IMREAD_GRAYSCALE)
    shape = image_array.shape + (1,)

    output = tf.argmax(predictions, axis=1)
    output = tf.reshape(output, shape)

    plt.figure(figsize=(15, 15))
    plt.imsave(save_loc, tf.keras.utils.array_to_img(output))
    plt.imshow(tf.keras.utils.array_to_img(output))


def main():
    np.random.seed(666)
    palettedata = []
    classifiers_list = [1]
    for _ in range(len(classifiers_list)):
        palettedata.extend(list(np.random.choice(range(256), size=3)))
        print(palettedata)
    num_entries_palette = 256
    num_bands = len("RGB")
    num_entries_data = len(palettedata) // num_bands
    palettedata.extend([0, 0, 0]
                       * (num_entries_palette
                          - num_entries_data))

    model_location = "model/saved_model/new_model.h5"
    save_location = "model/saved_model/img/tensorflow_modelv2.png"
    data_location = "model/full_classification.npy"
    original_location = "model/nucleus_color_cleaned.tif"

    model_to_tif(model_location, save_location, data_location, palettedata, original_location)

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)

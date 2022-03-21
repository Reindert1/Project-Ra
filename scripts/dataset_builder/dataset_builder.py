#!usr/bin/env python3

"""
Script to build dataset from given image and classifier
"""

__author__ = "Skippybal"
__version__ = "0.8"

import sys
import dataset_config as cfg

import PIL
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sklearn

PIL.Image.MAX_IMAGE_PIXELS = 268435460


def window_maker(array, window_size=(3, 3)):
    print("Rolling window")
    x_size = array.shape[0]
    y_size = array.shape[1]

    horizontal_pad = window_size[0] // 2
    vertical_pad = window_size[1] // 2
    array = np.pad(array, [(horizontal_pad, vertical_pad), (horizontal_pad, vertical_pad)],
                   mode='constant', constant_values=0)
    strides = array.strides * 2
    shape = (y_size, x_size) + window_size
    return np.lib.stride_tricks.as_strided(array, shape, strides)


def windows_to_dataset(windows):
    print("Windows to dataset")
    # array = None
    # for i in range(windows.shape[0]):
    #     for b in range(windows.shape[1]):
    #         if array is None:
    #             array = windows[i, b, :, :].flatten()
    #             array = np.expand_dims(array, axis=0)
    #         else:
    #             array = np.append(array, np.expand_dims(windows[i, b, :, :].flatten(), axis=0), axis=0)
    #print(windows.shape)

    axis0 = windows.shape[0] * windows.shape[1]
    axis1 = windows.shape[2] * windows.shape[3]
    array = windows.reshape(axis0, axis1)
    #print(array.shape)
    return array


def build_gaussian(image_array, image, layers=1):
    print("Building gaussian pyramid")
    layer = image.copy()
    gaus_np = image_array
    #gaus_np = np.array(layer).flatten().transpose()
    #gaus_np = gaus_np.reshape(gaus_np.shape[0], 1)

    for i in range(layers):
        layer = cv.pyrDown(layer)
        lay2 = layer.copy()
        for j in range(i + 1):
            lay2 = cv.pyrUp(lay2)
        #print(gaus_np.shape)
        #print(lay2.shape)
        gaus_np = np.append(gaus_np, np.expand_dims(lay2.flatten().transpose(), axis=1), axis=1)

    return gaus_np


def add_classifier(array, classifier_array):
    print("Adding classifier")
    classifier_array_flat = np.expand_dims(classifier_array.flatten().transpose(), axis=1)
    full_array = np.append(array, classifier_array_flat, axis=1)
    return full_array


def read_image(image_tif, classifier_tif):
    print("Reading images")
    image_array = cv.imread(image_tif, cv.IMREAD_GRAYSCALE)

    classifier = Image.open(classifier_tif)
    classifier_array = np.array(classifier)

    return image_array, classifier_array


def main():
    window_shape = cfg.window
    image, classifier = read_image(cfg.image_path, cfg.classifier_path)
    windows = window_maker(image, window_shape)

    data_array = windows_to_dataset(windows)
    data_array = build_gaussian(data_array, image, cfg.gaussian_layers)

    full_array = add_classifier(data_array, classifier)

    if cfg.remove_zero:
        full_array = np.delete(full_array, np.where(full_array[:, -1] == 0)[0], axis=0)

    print(full_array.shape)
    print(full_array[0, :])

    np.save(cfg.save_location, full_array)

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
#!usr/bin/env python3

"""
Script to build gaussian pyramids dataset for a given image
"""

__author__ = "Skippybal"
__version__ = "0.1"

import cv2 as cv
import numpy as np
import sys


def build_gaussian(image, save_location, layers=1):
    image_array = cv.imread(image, cv.IMREAD_GRAYSCALE)
    layer = image_array.copy()
    gaus_np = np.array(layer).flatten().transpose()
    gaus_np = gaus_np.reshape(gaus_np.shape[0], 1)

    for i in range(layers):
        layer = cv.pyrDown(layer)
        lay2 = layer.copy()
        for j in range(i + 1):
            lay2 = cv.pyrUp(lay2)

        gaus_np = np.append(gaus_np, np.expand_dims(lay2.flatten().transpose(), axis=1), axis=1)

    np.save(save_location, gaus_np)
    return 0


def main():
    build_gaussian(snakemake.input[0], snakemake.output[0], snakemake.params["gaussian_layers"])
    return 0


if __name__ == '__main__':
    with open(snakemake.log[0], "w") as log_file:
        sys.stderr = sys.stdout = log_file
        exitcode = main()
        sys.exit(exitcode)


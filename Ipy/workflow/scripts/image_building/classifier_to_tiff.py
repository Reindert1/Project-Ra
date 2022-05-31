#!usr/bin/env python3

"""
Script to segment a given image using a given classifier
"""

__author__ = "Skippybal"
__version__ = "0.1"

import math
import pickle
import random

import numpy as np
import sys
import h5py
import numpy as np
import cv2 as cv
import gc
from sklearn.model_selection import train_test_split
from PIL import Image


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def model_to_tif(model_file, save_loc, dataset_loc, palette, original_image_loc):

    loaded_model = pickle.load(open(model_file, 'rb'))
    #full_pred = loaded_model.predict(data)
    x_data = np.load(dataset_loc, mmap_mode="r")

    full_pred = []
    #full_pred = np.empty()
    for val_batch in validation_generator(x_data, 50000):
        #np.append(full_pred, loaded_model.predict(val_batch))\
        #print(val_batch)
        val_batch_normal = val_batch / 255
        pred = loaded_model.predict(np.array(val_batch_normal))
        #print(np.unique(pred))
        full_pred.extend(pred)

    full_pred = np.asarray(full_pred)
    image_array = cv.imread(original_image_loc, cv.IMREAD_GRAYSCALE)
    shape = image_array.shape
    del image_array
    gc.collect()
    full_pred = np.reshape(full_pred, shape)
    print(np.unique(full_pred))

    #full_pred = full_pred.reshape(16384, 16384)
    full_image = Image.fromarray(full_pred, mode="P")
    full_image.putpalette(palette)
    full_image.save(save_loc)


def main():
    np.random.seed(666)
    palettedata = []
    classifiers_list = snakemake.config["classifiers"]
    print(classifiers_list)
    for _ in range(len(classifiers_list)):
        palettedata.extend(list(np.random.choice(range(256), size=3)))
        print(palettedata)
    num_entries_palette = 256
    num_bands = len("RGB")
    num_entries_data = len(palettedata) // num_bands
    palettedata.extend([0, 0, 0]
                       * (num_entries_palette
                          - num_entries_data))

    model_to_tif(snakemake.input[0], snakemake.output[0], #snakemake.wildcard["classifier"],
                 snakemake.input[1], palettedata, snakemake.params["original_image_location"])

    return 0


if __name__ == '__main__':
    #with open(snakemake.log[0], "w") as log_file:
    #    sys.stderr = sys.stdout = log_file
        exitcode = main()
        sys.exit(exitcode)
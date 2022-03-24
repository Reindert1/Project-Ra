#!usr/bin/env python3

"""
Script to build dataset from given image and classifier
"""

__author__ = "Skippybal"
__version__ = "0.8"

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from joblib import parallel_backend
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
import pickle
import PIL
from PIL import Image
from pathlib import Path
import random


def model_to_tif(model_file, model_name, data, palette):

    Path("images").mkdir(parents=True, exist_ok=True)
    loaded_model = pickle.load(open(model_file, 'rb'))
    full_pred = loaded_model.predict(data).reshape(16384, 16384)
    full_image = Image.fromarray(full_pred, mode="P")
    full_image.putpalette(palette)
    full_image.save(f"images/{model_name}.tif")


def main():
    data_array = np.load(
        "/homes/kanotebomer/Documents/Thema11/Project-Ra/scripts/dataset_builder/total_classification.npy",
        allow_pickle=True)[:, :-1]
    palettedata = [0, 0, 0, 0, 0, 255]
    num_entries_palette = 256
    num_bands = len("RGB")
    num_entries_data = len(palettedata) // num_bands
    palettedata.extend(palettedata[:num_bands]
                       * (num_entries_palette
                          - num_entries_data))
    model_to_tif("models/SGD.sav", "SGD", data_array, palettedata)

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
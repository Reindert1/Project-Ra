#!usr/bin/env python3

"""
Script to run SVM classifier on a given dataset
"""

__author__ = "Skippybal"
__version__ = "0.1"

import random
import sys

import numpy.random
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
import copy
import numpy as np

import math
import pickle
import random

import sys

import cv2 as cv
import gc
from sklearn.model_selection import train_test_split
from PIL import Image


def batch_generator(instances, ys, indixes, batch_size):
    #for i in range(0, len(indixes), batch_size):
    for i in range(0, batch_size * 1, batch_size):
        yield instances[indixes[i:i+batch_size]], ys[indixes[i:i+batch_size]]


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def train_svm(x_train, x_val, y_train, y_val):
    # model = SVC(random_state=0)
    print(f"current: SVM")
    # model.fit(x_train, y_train)

    param_grid = {'kernel': ('poly', 'rbf'),
                  'gamma': ('scale', 'auto'),
                  'C': [1, 10, 100]}
    base_estimator = SVC(gamma='scale')
    model = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                             factor=3, min_resources="exhaust").fit(x_train, y_train)

    print(model.best_params_)

    y_pred = []
    for val_batch in validation_generator(x_val, 50000):
        y_pred.extend(model.predict(val_batch))
    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

    return model


def model_to_tif(model, x_data, palette, original_image_loc, save_loc):

    loaded_model = model[1]

    full_pred = []
    for val_batch in validation_generator(x_data, 50000):
        pred = loaded_model.predict(np.array(val_batch))
        full_pred.extend(pred)

    full_pred = np.asarray(full_pred)
    image_array = cv.imread(original_image_loc, cv.IMREAD_GRAYSCALE)
    shape = image_array.shape
    del image_array
    gc.collect()
    full_pred = np.reshape(full_pred, shape)
    print(np.unique(full_pred))

    full_image = Image.fromarray(full_pred, mode="P")
    full_image.putpalette(palette)
    full_image.save(save_loc)


def main():
    random.seed(42)
    numpy.random.seed(42)
    data = np.load("/local-fs/bachelor-students/2021-2022/Thema12/ra_data/small_test/dataset/full_classification.npy")
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.33,
                                                        random_state=42)

    # indexes = list(range(0, len(x_train)))
    # random.shuffle(indexes)

    indexes = {}

    for i in np.unique(data[:, -1]):
        #print(i)
        #idxs = np.where(data[:, -1] == i)[0]
        idxs = np.where(y_train == i)[0]
        #print(idxs[:10])
        indexes[i] = idxs

    random_data_idx = []

    for index in indexes:
        index_list = indexes[index]
        # print(index_list.shape)
        small = numpy.random.choice(index_list, 4000)

        # print(small.shape)
        # print(small[:10])
        random_data_idx.extend(small)

    print(len(random_data_idx))

    svms = []
    # print(indexes[:10])
    #
    # index = 0
    rand_x = x_train[random_data_idx]
    rand_y = y_train[random_data_idx]
    print(rand_x[:10])
    print(rand_y[:10])
    #
    svms.append((0, train_svm(rand_x, x_test, rand_y, y_test)))

    # for mini_batch_x, mini_batch_y in batch_generator(x_train, y_train, indexes, 1028):
    #     index += 1
    #     # print(mini_batch_y)
    #
    #     svms.append((index, train_svm(mini_batch_x, x_test, mini_batch_y, y_test)))

    np.random.seed(666)
    palettedata = []
    classifiers_list = [1]
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

    #model_to_tif(svms[0], data[:, :-1], palettedata, "/commons/Themas/Thema11/Giepmans/work/tmp/larger_data.tif",
    #             "/commons/Themas/Thema11/Giepmans/work/SVM_Run/pred_2.tif")
    model_to_tif(svms[0], data[:, :-1], palettedata, "/commons/Themas/Thema11/Giepmans/work/train_small_r4_c7.tif",
                 "/commons/Themas/Thema11/Giepmans/work/SVM_Run/pred_help.tif")



    # voting = VotingClassifier(estimators=svms, voting='hard')

    # lr = LogisticRegression()
    #
    # sclf = StackingClassifier(classifiers=svms,
    #                           meta_classifier=lr, fit_base_estimators=False)
    #
    # sclf.fit(x_train, y_train)
    #
    # print('accuracy:', np.mean(y_test == sclf.predict(x_test)))

    # y_pred = []
    # for val_batch in validation_generator(x_test, 50000):
    #     y_pred.extend(voting.predict(val_batch))
    #
    # print("Accuracy:", metrics.accuracy_score(y_train, y_test))




    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)

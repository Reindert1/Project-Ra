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


def test_trainer(name_model, model, x_train, x_val, y_train, y_val):
    with parallel_backend('threading', n_jobs=-1):
        print(f"current: {name_model}")
        model.fit(x_train, y_train)

        filename = f'/homes/kanotebomer/Documents/Thema11/Project-Ra/scripts/machine_learning/models/{name_model}.sav'
        pickle.dump(model, open(filename, 'wb'))

        y_pred = model.predict(x_val)
        print("Accuracy:", metrics.accuracy_score(y_val, y_pred))


def batch_generator(instances, ys, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i+batch_size], ys[i:i+batch_size]


def sgdclass(x_train, x_val, y_train, y_val, classes):
    model = SGDClassifier(loss='log')  # shuffle=True is useless here
    shuffledRange = range(x_train.shape[0])
    n_iter = 5
    for n in range(n_iter):
        print(n)
        # random.shuffle(shuffledRange)
        # shuffledX = [x_train[i] for i in shuffledRange]
        # shuffledY = [y_train[i] for i in shuffledRange]

        #full = list(zip(x_train, y_train))
        #random.shuffle(full)

        #xs_train, ys_train = zip(*full)

        for mini_batch_x, mini_batch_y in batch_generator(x_train, y_train, 10000):
            #with parallel_backend('threading', n_jobs=-1):
            model.partial_fit(mini_batch_x, mini_batch_y,
                             classes=classes)

    filename = f'/homes/kanotebomer/Documents/Thema11/Project-Ra/scripts/machine_learning/models/SGD_mit.sav'
    pickle.dump(model, open(filename, 'wb'))

    y_pred = model.predict(x_val)
    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))


def model_to_tif(model_file, model_name, data, palette):
    Path("images").mkdir(parents=True, exist_ok=True)
    loaded_model = pickle.load(open(model_file, 'rb'))
    full_pred = loaded_model.predict(data).reshape(16384, 16384)
    full_image = Image.fromarray(full_pred, mode="P")
    full_image.putpalette(palette)
    full_image.save(f"images/{model_name}.tif")


def main():
    #data_array = np.load("/homes/kanotebomer/Documents/Thema11/Project-Ra/scripts/dataset_builder/total_classification.npy", allow_pickle=True)
    data_array = np.load(
        "/tmp/ra_data/total_classification.npy",
        allow_pickle=True)
    print(data_array.shape)
    unique_colors_train = np.unique(data_array[:, -1])
    print(unique_colors_train)
    #data_array = np.unique(data_array, axis=0)
    #print(data_array.shape)

    models = {  # 'MultinomalNB': MultinomialNB(),
        #'Kmeans': MiniBatchKMeans(n_clusters=5, random_state=0),  # , batch_size=5000, max_iter=20),
        'GaussianNB': GaussianNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(min_samples_split=100, random_state=18),
        #'SGDClassifier': SGDClassifier(max_iter=100, tol=1e-3, random_state=0)
    }
    #
    x_train, x_val, y_train, y_val = train_test_split(data_array[:, :-1], data_array[:, -1], test_size=0.3,
                                                      random_state=0)
    #
    for name, model in models.items():
        test_trainer(name, model, x_train, x_val, y_train, y_val)
    #sgdclass(x_train, x_val, y_train, y_val, unique_colors_train)

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
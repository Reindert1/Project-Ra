#!usr/bin/env python3

"""
Script to run ZeroR on given dataset
"""

__author__ = "Skippybal"
__version__ = "0.1"

import pickle
import sys
import h5py

from sklearn import metrics
from sklearn.dummy import DummyClassifier


def train_dummy(x_train, x_val, y_train, y_val, save_loc):
    """
    Train DummyClassifier, also know as ZeroR, on given data
    :param x_train: features of the training data
    :param x_val: features of the validation data
    :param y_train: labels of the training data
    :param y_val: labels of the validation data
    :param save_loc: location to save classifier
    :return: 0
    """
    model = DummyClassifier(strategy="most_frequent", random_state=42)
    print(f"current: ZeroR")
    model.fit(x_train, y_train)

    pickle.dump(model, open(save_loc, 'wb'))

    y_pred = model.predict(x_val)
    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

    return 0


def main():
    """
    Main function
    """

    hdf5_file = snakemake.input[0]
    save_location = snakemake.output[0]

    f = h5py.File(hdf5_file, 'r')
    x_train = f.get('x_train')
    y_train = f.get('y_train')
    x_val = f.get('x_test')
    y_val = f.get('y_test')

    train_dummy(x_train, x_val, y_train, y_val, save_location)

    f.close()

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)

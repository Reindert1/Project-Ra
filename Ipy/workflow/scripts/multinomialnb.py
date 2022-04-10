#!usr/bin/env python3

"""
Script to run MultinomialNB on given dataset
"""

__author__ = "Skippybal"
__version__ = "0.1"

import pickle
import random
import sys
import h5py
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


def batch_generator(instances, ys, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i+batch_size], ys[i:i+batch_size]


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def train_bayes(x_train, x_val, y_train, y_val, save_loc, metric_loc):
    classes = np.unique(y_train)
    model = MultinomialNB()
    #with parallel_backend('threading', n_jobs=-1):
    metric_dict = {"Model": "MultinomialNB"}
    print(f"current: MultinomialNB")
    #range_x = list(range(len(x_train)))
    n_iter = 5
    for n in range(n_iter):
        print(n)
        # random.shuffle(range_x)
        # print(range_x[:5])
        for mini_batch_x, mini_batch_y in batch_generator(x_train, y_train, 1000):
            # with parallel_backend('threading', n_jobs=-1):
            model.partial_fit(mini_batch_x, mini_batch_y,
                              classes=classes)

    #filename = f'models/{name_model}.sav'
    pickle.dump(model, open(save_loc, 'wb'))

    y_pred = []
    for val_batch in validation_generator(x_val, 50000):
        y_pred.extend(model.predict(val_batch))
    accuracy = metrics.accuracy_score(y_val, y_pred)
    metric_dict["Accuracy"] = accuracy
    pickle.dump(metric_dict, open(metric_loc, 'wb'))
    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

    return 0


def main():
    #model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)
    hdf5_file = snakemake.input[0]
    save_location = snakemake.output["model"]
    metric_loc = snakemake.output["metrics"]

    f = h5py.File(hdf5_file, 'r')
    x_train = f.get('x_train')
    y_train = f.get('y_train')
    x_val = f.get('x_test')
    y_val = f.get('y_test')

    # x_train = np.load(f"{snakemake.config['dataset_dir']}dataset/x_train.npy", mmap_mode="r")
    # y_train = np.load(f"{snakemake.config['dataset_dir']}dataset/y_train.npy", mmap_mode="r")
    # x_val = np.load(f"{snakemake.config['dataset_dir']}dataset/x_test.npy", mmap_mode="r")
    # y_val = np.load(f"{snakemake.config['dataset_dir']}dataset/y_test.npy", mmap_mode="r")

    #print(x_val.shape)
    #print(classes)
    train_bayes(x_train, x_val, y_train, y_val, save_location, metric_loc)

    f.close()

    return 0


if __name__ == '__main__':
    with open(snakemake.log[0], "w") as log_file:
        sys.stderr = sys.stdout = log_file
        exitcode = main()
        sys.exit(exitcode)
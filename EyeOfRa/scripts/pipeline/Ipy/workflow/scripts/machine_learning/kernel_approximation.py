#!usr/bin/env python3

"""
Script to run kernel_approximation on a given dataset making use of bayesian hyperparameter tuning
"""

__author__ = "Skippybal"
__version__ = "0.1"

import pickle
import numpy as np
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn import metrics, pipeline
import h5py


def batch_generator(instances, ys, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i+batch_size], ys[i:i+batch_size]


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def train_sgd(x_train, x_val, y_train, y_val, classes, optimal_params, save_loc, metric_loc, class_weights):
    model = SGDClassifier(random_state=0, loss="log")

    rbf_feature = RBFSampler(gamma=1, random_state=1)

    # rbf_feature.fit(x_train)
    # x_train = rbf_feature.transform(x_train)
    # x_val = rbf_feature.transform(x_val)

    # approx_svm = pipeline.Pipeline(
    #     [("feature_map", rbf_feature), ("sgd", SGDClassifier(random_state=0, loss="log"))]
    # )

    n_iter = 10
    for n in range(n_iter):
        print(n)

        for mini_batch_x, mini_batch_y in batch_generator(x_train, y_train, 1028):
            #transformed_x = rbf_feature.fit_transform(mini_batch_x)
            approx_svm.partial_fit(mini_batch_x, mini_batch_y,
                              classes=classes)

    pickle.dump(model, open(save_loc, 'wb'))

    y_pred = []
    for val_batch in validation_generator(x_val, 10000):
        y_pred.extend(model.predict(val_batch))

    metric_dict = {"Model": "Kernel approximation"}
    accuracy = metrics.accuracy_score(y_val, y_pred)
    #metric_dict["Accuracy"] = accuracy
    confus_matrix = metrics.confusion_matrix(y_val, y_pred)
    roc = metrics.roc_curve(y_val, y_pred)
    roc_auc_curve = metrics.roc_auc_score(y_val, y_pred)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_val, y_pred)

    metric_dict["Accuracy"] = accuracy
    metric_dict["confus_matrix"] = confus_matrix
    metric_dict["roc"] = roc
    metric_dict["roc_auc_curve"] = roc_auc_curve
    metric_dict["balanced_accuracy_score"] = balanced_accuracy_score

    pickle.dump(metric_dict, open(metric_loc, 'wb'))

    print("Accuracy: ", accuracy)

    return 0


def main():
    hdf5_file = snakemake.input[0]
    save_location = snakemake.output["model"]
    metric_loc = snakemake.output["metrics"]

    f = h5py.File(hdf5_file, 'r')
    x_train = f.get('x_train')
    y_train = f.get('y_train')
    x_val = f.get('x_test')
    y_val = f.get('y_test')
    classes = np.unique(y_train)

    # class_weights = compute_class_weight(class_weight='balanced',
    #                                      y=y_train, classes=classes)
    # class_weights = dict(zip(np.unique(classes), class_weights))
    #print(x_val.shape)
    #print(classes)
    #optimal_params = find_optimal(x_train, x_val, y_train, y_val, classes, class_weights)
    optimal_params = None
    class_weights = None
    train_sgd(x_train, x_val, y_train, y_val, classes, optimal_params, save_location, metric_loc, class_weights)

    f.close()

    return 0


if __name__ == '__main__':
    # with open(snakemake.log[0], "w") as log_file:
    #     sys.stderr = sys.stdout = log_file
        exitcode = main()
        sys.exit(exitcode)

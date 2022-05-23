#!usr/bin/env python3

"""
Script to run SVM classifier on a given dataset
"""

__author__ = "Skippybal"
__version__ = "0.1"

import pickle
import sys
from sklearn import metrics
from sklearn.svm import SVC

import h5py


def batch_generator(instances, ys, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i+batch_size], ys[i:i+batch_size]


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def train_svm(x_train, x_val, y_train, y_val, save_loc, metric_loc):
    model = SVC(random_state=0)
    metric_dict = {"Model": "SVM"}
    print(f"current: SVM")
    model.fit(x_train, y_train)

    pickle.dump(model, open(save_loc, 'wb'))

    y_pred = []
    for val_batch in validation_generator(x_val, 50000):
        y_pred.extend(model.predict(val_batch))
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

    train_svm(x_train, x_val, y_train, y_val, save_location, metric_loc)

    f.close()

    return 0


if __name__ == '__main__':
    with open(snakemake.log[0], "w") as log_file:
        sys.stderr = sys.stdout = log_file
        exitcode = main()
        sys.exit(exitcode)

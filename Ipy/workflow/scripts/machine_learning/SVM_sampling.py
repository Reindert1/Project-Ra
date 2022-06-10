#!usr/bin/env python3

"""
Script to run SVM classifier on a given dataset
"""

__author__ = "Skippybal"
__version__ = "0.1"

import pickle
import random
import sys
import numpy as np
import numpy.random
from sklearn import metrics
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.svm import SVC
# from tune_sklearn import TuneSearchCV
# from ray import tune


def batch_generator(instances, ys, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size], ys[i:i + batch_size]


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def train_svm(x_train, x_val, y_train, y_val, save_loc, metric_loc):
    model_base = SVC(random_state=0)
    metric_dict = {"Model": "SVM_sampling"}
    x_train = x_train / 255
    print(f"current: SVM_sampling")

    model = model_base

    # param_dists = {
    #     'loss': tune.choice(['rbf', 'poly']),
    #     'gamma': tune.choice(['gamma', 'auto']),
    #     'C': tune.uniform(0.1, 10),
    # }
    #
    # model = TuneSearchCV(model_base,
    #                      param_distributions=param_dists,
    #                      n_trials=2,
    #                      early_stopping=True,  # uses Async HyperBand if set to True
    #                      max_iters=10,
    #                      search_optimization="hyperopt"
    #                      )

    param_grid = {'kernel': ('poly', 'rbf'),
                  'gamma': ('scale', 'auto'),
                  'C': [1, 10, 100]}
    #base_estimator = SVC(gamma='scale')
    base_estimator = SVC()
    model = HalvingGridSearchCV(base_estimator, param_grid, cv=5,
                             factor=3, min_resources="exhaust").fit(x_train, y_train)

    print(model.best_params_)

    #model.fit(x_train, y_train)

    #pickle.dump(model, open(save_loc, 'wb'))

    y_train_pred = []
    for val_batch in validation_generator(x_train, 50000):
        val_batch_normal = val_batch
        y_train_pred.extend(model.predict(val_batch_normal))
    accuracy = metrics.accuracy_score(y_train, y_train_pred)
    metric_dict["Train_Accuracy"] = accuracy

    y_pred = []
    for val_batch in validation_generator(x_val, 50000):
        val_batch_normal = val_batch / 255
        y_pred.extend(model.predict(val_batch_normal))
    accuracy = metrics.accuracy_score(y_val, y_pred)
    # metric_dict["Accuracy"] = accuracy

    confus_matrix = metrics.confusion_matrix(y_val, y_pred)
    # roc = metrics.roc_curve(y_val, y_pred)
    # roc_auc_curve = metrics.roc_auc_score(y_val, y_pred)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_val, y_pred)

    metric_dict["Test_Accuracy"] = accuracy
    metric_dict["confus_matrix"] = confus_matrix
    # metric_dict["roc"] = roc
    # metric_dict["roc_auc_curve"] = roc_auc_curve
    metric_dict["balanced_accuracy_score"] = balanced_accuracy_score

    pickle.dump(metric_dict, open(metric_loc, 'wb'))
    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

    return model


def main():
    numpy_file = snakemake.input[0]
    save_location = snakemake.output["model"]
    metric_loc = snakemake.output["metrics"]

    random.seed(42)
    numpy.random.seed(42)
    data = np.load(numpy_file)
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.33,
                                                        random_state=42)
    indexes = {}
    svms = []

    for svm_index in range(5):

        for i in np.unique(data[:, -1]):
            idxs = np.where(y_train == i)[0]
            indexes[i] = idxs

        random_data_idx = []

        for index in indexes:
            index_list = indexes[index]
            small = numpy.random.choice(index_list, 500)

            random_data_idx.extend(small)

        print(len(random_data_idx))


        rand_x = x_train[random_data_idx]
        rand_y = y_train[random_data_idx]
        print(rand_x[:10])
        print(rand_y[:10])

        # svms[svm_index] = train_svm(rand_x, x_test, rand_y, y_test, save_location, metric_loc)
        svms.append(train_svm(rand_x, x_test, rand_y, y_test, save_location, metric_loc))
        #print(svms)

    pickle.dump(svms, open(save_location, 'wb'))
    print(svms)

    return 0


if __name__ == '__main__':
    # with open(snakemake.log[0], "w") as log_file:
    #     sys.stderr = sys.stdout = log_file
    exitcode = main()
    sys.exit(exitcode)

import pickle

import numpy as np
import sys
from dask.distributed import Client, LocalCluster
import dask.array as da
from dask_ml.model_selection import HyperbandSearchCV
from sklearn.linear_model import SGDClassifier
from dask_ml.model_selection import train_test_split
from scipy.stats import uniform, loguniform
from sklearn import metrics
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
import h5py


def batch_generator(instances, ys, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i+batch_size], ys[i:i+batch_size]


def validation_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i + batch_size]


def find_optimal(x_train, x_val, y_train, y_val, classes):
    #model = SGDClassifier(loss='log')  # shuffle=True is useless here
    #model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)

    # params = {'alpha': loguniform(1e-2, 1e0),
    #           'l1_ratio': uniform(0, 1)}

    space = [Real(10 ** -5, 10 ** 0, "log-uniform", name='alpha'),
             Real(0, 1, name='l1_ratio')]

    @use_named_args(space)
    def objective(**params):
        model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)
        model.set_params(**params)

        n_iter = 1
        for n in range(n_iter):
            print(n)

            for mini_batch_x, mini_batch_y in batch_generator(x_train, y_train, 50000):
                # with parallel_backend('threading', n_jobs=-1):
                model.partial_fit(mini_batch_x, mini_batch_y,
                                  classes=classes)

        y_pred = []
        for val_batch in validation_generator(x_val, 50000):
            y_pred.extend(model.predict(val_batch))
        accurary = metrics.accuracy_score(y_val, y_pred)
        loss = 1 - accurary
        print("Accuracy: ", accurary)

        return loss


    # filename = f'/homes/kanotebomer/Documents/Thema11/Project-Ra/scripts/machine_learning/models/SGD_mit.sav'
    # pickle.dump(model, open(filename, 'wb'))

    res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)
    print(res_gp.fun)
    print(res_gp.x)

    optimal_params = {'alpha': res_gp.x[0],
                      'l1_ratio': res_gp.x[1]}
    print(optimal_params)
    return optimal_params


def train_sgd(x_train, x_val, y_train, y_val, classes, optimal_params, save_loc):
    model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0,
                          alpha=optimal_params["alpha"], l1_ratio=optimal_params["l1_ratio"])

    n_iter = 1
    for n in range(n_iter):
        print(n)

        for mini_batch_x, mini_batch_y in batch_generator(x_train, y_train, 10000):
            # with parallel_backend('threading', n_jobs=-1):
            model.partial_fit(mini_batch_x, mini_batch_y,
                              classes=classes)

    pickle.dump(model, open(save_loc, 'wb'))

    y_pred = []
    for val_batch in validation_generator(x_val, 10000):
        y_pred.extend(model.predict(val_batch))
    accurary = metrics.accuracy_score(y_val, y_pred)

    print("Accuracy: ", accurary)

    return 0


def main():
    #model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)
    hdf5_file = snakemake.input[0]
    save_location = snakemake.output[0]

    f = h5py.File(hdf5_file, 'r')
    x_train = f.get('x_train')
    y_train = f.get('y_train')
    x_val = f.get('x_test')
    y_val = f.get('y_test')
    classes = np.unique(y_train)
    #print(x_val.shape)
    #print(classes)
    optimal_params = find_optimal(x_train, x_val, y_train, y_val, classes)
    train_sgd(x_train, x_val, y_train, y_val, classes, optimal_params, save_location)

    f.close()

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
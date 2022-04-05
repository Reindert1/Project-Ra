import numpy as np
import sys
from dask.distributed import Client, LocalCluster
import dask.array as da
from dask_ml.model_selection import HyperbandSearchCV
from sklearn.linear_model import SGDClassifier
from dask_ml.model_selection import train_test_split
from scipy.stats import uniform, loguniform


def concatter(data_location, classifier_list, temp_file, temp_file2):

    return 0


def main():
    #cluster = LocalCluster()
    #client = Client(n_workers=2, threads_per_worker=1)
    client = Client(n_workers=2, threads_per_worker=1)
    print(client)

    array = np.load("/Thema11/Ipy/results/dataset/full_classification.npy", mmap_mode="r")
    print(array.shape)
    print(array[0])
    x = da.from_array(array[:, :-1], chunks=10000)
    y = da.from_array(array[:, -1], chunks=10000)
    print(x.shape)
    print(y.shape)
    print(x)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    clf = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)

    params = {'alpha': loguniform(1e-2, 1e0),
              'l1_ratio': uniform(0, 1)}

    n_examples = 2 * len(x_train)
    n_params = 2 #8

    max_iter = n_params
    chunks = n_examples // n_params

    print(max_iter, chunks)

    x_train2 = da.rechunk(x_train, chunks=chunks)
    y_train2 = da.rechunk(y_train, chunks=chunks)
    print(x_train2)

    search = HyperbandSearchCV(clf, params, max_iter=max_iter, random_state=0, patience=True)
    print(search.metadata["partial_fit_calls"])

    classes = da.unique(y_train).compute()
    print(classes)

    search.fit(x_train2, y_train2, classes=classes)
    print(search.best_params_)

    search.score(x_test, y_test)

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
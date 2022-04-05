import math
import pickle
import random

import numpy as np
import sys
import h5py
import numpy as np
import cv2 as cv
import gc
from sklearn.model_selection import train_test_split
from PIL import Image
from joblib import parallel_backend
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB



def train_bayes(x_train, x_val, y_train, y_val, save_loc):
    model = GaussianNB()
    with parallel_backend('threading', n_jobs=-1):
        print(f"current: GaussianNB")
        model.fit(x_train, y_train)

        #filename = f'models/{name_model}.sav'
        pickle.dump(model, open(save_loc, 'wb'))

        y_pred = model.predict(x_val)
        print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

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
    #print(x_val.shape)
    #print(classes)
    train_bayes(x_train, x_val, y_train, y_val, save_location)

    f.close()

    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
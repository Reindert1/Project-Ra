#!usr/bin/env python3

"""
    Boosting algorithm using Adaboost
        - runs sklearn adaboost ensemble learner
        - trains on training data and runs validation
        - exports model
        - visualizes model
"""

__author__ = "Reindert1"
__version__ = 0.1


# Imports
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from PIL import Image


class AdaBoost:
    def __init__(self, data_array, dataset):
        self.data_array = data_array
        self.dataset = dataset

        self.model = None
        self.x_val = None

        AdaBoost.fit(self)
        AdaBoost.export_model(self)

    def fit(self):
        x_train, x_val, y_train, y_val = train_test_split(self.data_array[:, :-1], self.data_array[:, -1], random_state=0)
        self.x_val = x_val
        print("Initializing booster...")
        model = AdaBoostClassifier(n_estimators=5, learning_rate=0.1, algorithm='SAMME')

        print("Fitting training data...")
        model.fit(x_train, y_train)

        prediction = model.score(x_train, y_train)

        print('The accuracy on training data is: ', prediction * 100, '%')

        print("Running validation data...")
        model.predict(x_val)

        prediction = model.score(x_val, y_val)
        print('The accuracy on validation data is: ', prediction * 100, '%')

        # Setting model
        self.model = model
        return self

    def export_model(self):
        print("Pickle dump")
        pickle.dump(self.model, open("Booster.sav", 'wb'))
        AdaBoost.model_to_tif(self)

    def model_to_tif(self):
        palettedata = [0, 0, 0, 0, 0, 255]
        num_entries_palette = 256
        num_bands = len("RGB")
        num_entries_data = len(palettedata) // num_bands
        palettedata.extend(palettedata[:num_bands]
                           * (num_entries_palette
                              - num_entries_data))

        loaded_model = self.model
        full_pred = loaded_model.predict(self.x_val).reshape(16384, 16384)
        full_image = Image.fromarray(full_pred, mode="P")
        full_image.putpalette(palettedata)
        full_image.save("Booster.tif")


def main():
    data_array = np.load("r4-c7_nucleus.npy", allow_pickle=True)
    print("Shape:", data_array.shape)

    dataset = pd.DataFrame(data_array, columns=['Pixel_val1', 'Pixel_val2', 'Pixel_val3', 'Pixel_val4', 'Pixel_val5',
                                                'Pixel_val6', 'Pixel_val7', 'Pixel_val8', 'Pixel_val9', 'Gauss1',
                                                'Gauss2', 'Gauss3', 'Gauss4', 'Label'])

    AdaBoost(data_array, dataset)

    return 0


if __name__ == "__main__":
    exitcode = main()
    sys.exit(exitcode)


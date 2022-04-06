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
import booster2 as model2
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
        x_train, x_val, y_train, y_val = train_test_split(self.data_array[:, :-1],
                                                          self.data_array[:, -1],
                                                          random_state=42)
        self.x_val = x_val
        print("Initializing booster...")
        model = AdaBoostClassifier(n_estimators=10, learning_rate=0.1, algorithm='SAMME')

        print("Fitting training data...")
        model.fit(x_train, y_train)

        # Setting model
        self.model = model
        return self

    def export_model(self):
        print("Pickle dump")
        pickle.dump(self.model, open("Booster.sav", 'wb'))
        AdaBoost.model_to_tif(self)

    def model_to_tif(self):
        data_array = np.load("r4-c7_nucleus.npy", allow_pickle=True)[:, :-1]
        palettedata = []
        num_entries_palette = 256
        for _ in range(1):
            palettedata.extend(list(np.random.choice(range(256), size=3)))
            print(palettedata)
        num_bands = len("RGB")
        num_entries_data = len(palettedata) // num_bands
        palettedata.extend([0, 0, 0]
                           * (num_entries_palette
                              - num_entries_data))
        save_image("Booster.sav",
                     data_array, palettedata)

    def save_image(self, data, palette):
        loaded_model = pickle.load(model2)
        full_pred = loaded_model.predict(data).reshape(16384, 16384)
        print(np.unique(full_pred))
        full_image = Image.fromarray(full_pred, mode="P")
        full_image.putpalette(palette)
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


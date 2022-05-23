import numpy as np
import h5py

from keras import layers
from tensorflow import keras

DATA_PATH = "../../data/stripes4/"

f = h5py.File(DATA_PATH + "dataset/train.h5py", "r")
x_train = f.get('x_train')
y_train = f.get('y_train')
x_val = f.get('x_test')
y_val = f.get('y_test')

model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
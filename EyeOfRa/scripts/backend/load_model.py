# import tensorflow as tf
import h5py

path = "../../resources/model_history/trained_model.h5"
file = h5py.File(path, "r")
print(file.keys())
print(file["model_weights"])
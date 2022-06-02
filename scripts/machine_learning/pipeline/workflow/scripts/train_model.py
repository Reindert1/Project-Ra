import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pathlib

from model import create_model

METRICS = [
      "accuracy"
]



def load_data(nyp_path):
    path = pathlib.Path(nyp_path)
    if path.exists() and path.suffix == ".npy":
        data = np.load(nyp_path, allow_pickle=True)
        return data
    else:
        if not path.exists():
            raise FileNotFoundError
        else:
            raise ValueError("Only '.npy' files are accepted")


def save_model(save_path, model):
    path = pathlib.Path(save_path)
    if path.is_dir():
        path = path / "new_model.h5"

    model.save(path)


def train_model(data_path, model_save_path, model=None,
                early_stopping=True, log_training="history.cvs"):
    if type(data_path) is not str:
        raise TypeError("'data_path' should be a string!")

    if type(model_save_path) is not str:
        raise TypeError("'model_save_path' should be a string!")

    if type(log_training) is not str:
        raise TypeError("'data_path' should be a string!")

    data = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], random_state=69)

    callbacks = []
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4))

    if log_training:
        callbacks.append(tf.keras.callbacks.CSVLogger(log_training, separator=",", append=True))

    print("Fit model on training data")
    history = model.fit(x_train, y_train, epochs=150, validation_split=0.25,
                        callbacks=callbacks)


    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=150)
    print("test loss, test acc:", results)

    print("saving model!")
    save_model(model_save_path, model)

    return 1


if __name__ == '__main__':
    # TODO Callback toggling
    model: tf.keras.Sequential = create_model((109,), metrics=METRICS)
    print(f"Input: {snakemake.input[0]}")
    print(f"Model output: {snakemake.output[0]}")
    print(f"Early stopping: {snakemake.params['early_stopping']}")

    trained_model = train_model(snakemake.input[0], snakemake.output[0],
                                model=model, early_stopping=snakemake.params["early_stopping"], log_training="history.csv")

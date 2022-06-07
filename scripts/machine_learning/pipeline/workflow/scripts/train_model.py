import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pathlib
import os

from model import create_model

METRICS = [
    "accuracy",
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

    print("[INFO]\tFit model on training data")
    history = model.fit(x_train, y_train, epochs=150, validation_split=0.25,
                        callbacks=callbacks)

    print("[INFO]\tEvaluating on test data")
    results = model.evaluate(x_test, y_test, batch_size=150)
    print("[INFO]\ttest loss, test acc:", results)

    print("[INFO]\tsaving model!")
    save_model(model_save_path, model)

    return 1


if __name__ == '__main__':
    history_path = snakemake.output[1]
    input_path = snakemake.input[0]
    model_output_path = snakemake.output[0]

    # Remove history.csv
    if os.path.exists(history_path):
        os.remove(history_path)

    # Create model
    model: tf.keras.Sequential = create_model((109,), metrics=METRICS)

    print(f"\n[INFO]\tInput data: {pathlib.Path(input_path)}")
    print(f"[INFO]\tModel output: {pathlib.Path(model_output_path)}")
    print(f"[INFO]\tEarly stopping: {snakemake.params['early_stopping']}\n")

    trained_model = train_model(input_path, model_output_path,
                                model=model, early_stopping=snakemake.params["early_stopping"],
                                log_training=history_path)

import tensorflow as tf
import numpy as np

import sys
import os

from model import create_model


def get_input(nyp_path):
    data = np.load(nyp_path)

if __name__ == '__main__':
    model: tf.keras.Sequential = create_model((109,))


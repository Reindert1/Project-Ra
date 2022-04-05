import cv2 as cv
import numpy as np
import sys


def window_maker(array, window_size=(3, 3)):
    x_size = array.shape[0]
    y_size = array.shape[1]

    horizontal_pad = window_size[0] // 2
    vertical_pad = window_size[1] // 2
    array = np.pad(array, [(horizontal_pad, vertical_pad), (horizontal_pad, vertical_pad)],
                   mode='constant', constant_values=0)
    strides = array.strides * 2
    shape = (y_size, x_size) + window_size
    return np.lib.stride_tricks.as_strided(array, shape, strides)


def windows_to_dataset(windows, outfile):
    axis0 = windows.shape[0] * windows.shape[1]
    axis1 = windows.shape[2] * windows.shape[3]
    array = windows.reshape(axis0, axis1)
    np.save(outfile, array)
    return 0


def main():
    image = snakemake.input[0]
    window_shape = snakemake.params["window_size"]
    image_array = cv.imread(image, cv.IMREAD_GRAYSCALE)
    windows = window_maker(image_array, window_shape)
    windows_to_dataset(windows, snakemake.output[0])


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
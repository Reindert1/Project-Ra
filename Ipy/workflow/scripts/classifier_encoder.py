import cv2 as cv
import numpy as np
import sys
import PIL
from PIL import Image
from numpy.lib.format import open_memmap
import gc
import os

PIL.Image.MAX_IMAGE_PIXELS = 268435460


def concatter(data_location, classifier_list, temp_file, temp_file2):
    rows = 0
    cols = len(classifier_list)
    dtype = None
    #with data_location[0] as data_file:
    data_file = data_location[0]
    print(data_file)
    #data = np.load(data_file, mmap_mode='r', allow_pickle=True)
    classifier = Image.open(data_file)
    classifier = np.array(classifier, dtype=np.uint8)
    classifier = np.expand_dims(classifier.flatten().transpose(), axis=1)
    rows = classifier.shape[0]
    dtype = classifier.dtype
    print(dtype)

    del classifier

    print(cols)
    print(rows)
    merged = np.memmap(temp_file, dtype=np.uint8, mode='w+', shape=(rows, cols))
    data_index = 0
    for data_file in data_location:
        classifier = Image.open(data_file)
        classifier_array = np.array(classifier, dtype=np.uint8)
        classifier_array_flat = classifier_array.flatten().transpose()

        merged[:, data_index] = classifier_array_flat
        data_index += 1
    #merged.flush()

    disk_array = open_memmap(temp_file2, mode='w+', dtype=merged.dtype, shape=merged.shape)
    disk_array[:] = merged[:]
    del merged
    gc.collect()
    os.remove(temp_file)

    return 0


def decoder(temp_file, outfile):
    array = np.load(temp_file, mmap_mode='c')
    print(array.dtype)

    #full_class = np.memmap(outfile, dtype=dtype, mode='w+', shape=(nrows, 1))

    artificial_zero = np.all(array[:, :] == 0, axis=1).astype(np.uint8)
    artificial_zero = np.expand_dims(artificial_zero, axis=1)
    data_array = np.append(array, artificial_zero, axis=1)
    print(data_array.dtype)

    #split_column = labels_columns - 1
    labels = np.expand_dims(np.argmax(data_array[:, :], axis=1), axis=1).astype(np.uint8)
    print(labels.dtype)
    np.save(outfile, labels)
    #data_array = np.delete(data_array, np.s_[split_column:], axis=1)
    #data_array = np.append(data_array, labels, axis=1).astype(np.uint8)

    del array
    gc.collect()
    os.remove(temp_file)


def main():
    concatter(snakemake.input, snakemake.params["classifiers"], snakemake.params["temp_memmap"],
              snakemake.params["temp_memmap_npy"])
    decoder(snakemake.params["temp_memmap_npy"], snakemake.output["full"])
    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
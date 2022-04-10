#!usr/bin/env python3

"""
Script to split dataset into training and testing data
"""

__author__ = "Skippybal"
__version__ = "0.1"

import math
import random

import numpy as np
import sys
import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def batch_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i+batch_size, :] #instances[i:i+batch_size, :-1] , instances[i:i+batch_size, -1]

# def batch_generator(instances, batch_size, shuffle_dex):
#     for i in range(0, len(shuffle_dex), batch_size):
#         yield instances[shuffle_dex[i:i+batch_size], :] #instances[i:i+batch_size, :-1] , instances[i:i+batch_size, -1]


def batch_index_generator(instances, batch_size):
    for i in range(0, len(instances), batch_size):
        yield instances[i:i+batch_size]


def split_data(memmap_file, h5py_file):
    all_data = np.load(memmap_file, mmap_mode="r")
    print(all_data.shape[0])
    print(all_data.dtype)

    x_ids = list(range(all_data.shape[0]))
    print(1)
    x_train_ids, x_test_ids = train_test_split(x_ids, test_size=0.33, random_state=42)
    print("Done indexes")
    #np.save("/Thema11/Ipy/results/dataset/train.npy", all_data[x_train_ids])
    #np.save("/Thema11/Ipy/results/dataset/test.npy", all_data[x_test_ids])
    # cut_loc = math.floor(x_ids[-1] * 0.6)
    # x_train_ids = np.random.shuffle(x_ids)[cut_loc:]

    #idx = np.random.choice(range(x_ids[-1]), x_ids[-1])
    #print(idx[:10])

    # print(x_train_ids[:10])
    #print(Y_train[:10])

    generator = batch_generator(all_data, 10000)
    #generator = batch_generator(all_data, 10000, x_train_ids)
    # batch_x, batch_y = next(generator)
    batch_x = next(generator)
    dtype_x = batch_x.dtype
    #dtype_y = batch_y.dtype
    row_count = batch_x.shape[0]

    generator2 = batch_index_generator(x_train_ids, 10000)


    f = h5py.File('/Thema11/Ipy/results/dataset/train.h5py', 'w', libver='latest')
    maxshape_x = (None,) + batch_x.shape[1:]
    # maxshape_y = (None,) + batch_y.shape[1:]
    features = f.create_dataset(f"features", shape=batch_x.shape, maxshape=maxshape_x, chunks=batch_x.shape,
                                dtype=dtype_x)
    # labels = f.create_dataset(f"labels", shape=batch_y.shape, maxshape=maxshape_y, chunks=batch_y.shape,
    #                           dtype=dtype_y)

    print("File build")

    #labels = f.create_dataset(f"labels", shape=np.uint8, data=all_data[x_train_ids, -1])

    # features = f.create_dataset(f"features", data=all_data[x_train_ids, :-1], dtype=np.uint8)
    # labels = f.create_dataset(f"labels", data=all_data[x_train_ids, -1], dtype=np.uint8)
    # features = f.create_dataset(f"features", dtype=np.uint8, shape=(len(x_train_ids), all_data.shape[1]),
    #                            chunks=(10, 10, 10, 2048),compression=32001,compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
    #labels = f.create_dataset(f"labels", dtype=np.uint8)

    #for batch_x, batch_y in generator:
    for batch_x in generator:
        print("Adding index")
        features.resize(row_count + batch_x.shape[0], axis=0)
        print("Adding_data")
        features[row_count:] = batch_x

        # labels.resize(row_count + batch_y.shape[0], axis=0)
        # labels[row_count:] = batch_y

        row_count += batch_x.shape[0]

    # for indexes in generator2:
    #     print("Adding index")
    #     features.resize(row_count + batch_x.shape[0], axis=0)
    #     print("Adding_data")
    #     features[row_count:] = all_data[indexes, :]
    #     #features[row_count:] = all_data[indexes, :-1]
    #
    #     # labels.resize(row_count + batch_y.shape[0], axis=0)
    #     # labels[row_count:] = all_data[indexes, -1]
    #
    #     row_count += batch_x.shape[0]

    f.close()

    # f = h5py.File('/Thema11/Ipy/results/dataset/test.h5py', 'w')
    # f.create_dataset(f"features", data=all_data[x_test_ids, :-1], dtype=np.uint8)
    # f.create_dataset(f"labels", data=all_data[x_test_ids, -1], dtype=np.uint8)
    # f.close()
    return 0


def split_test2(memmap_file, h5py_file):
    #all_data = np.load(memmap_file, mmap_mode="r")
    all_data = np.load(memmap_file)
    x_train, x_test, y_train, y_test = train_test_split(all_data[:, :-1], all_data[:, -1], test_size=0.33, random_state=42)
    print("Done indexes")

    f = h5py.File(h5py_file, 'w', libver='latest')
    f.create_dataset('x_train', data=x_train)
    print("x_train in file")
    f.create_dataset('x_test', data=x_test)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('y_test', data=y_test)
    f.close()

    # np.save(f"{snakemake.config['dataset_dir']}dataset/x_train.npy", x_train)
    # np.save(f"{snakemake.config['dataset_dir']}dataset/x_test.npy", x_test)
    # np.save(f"{snakemake.config['dataset_dir']}dataset/y_train.npy", y_train)
    # np.save(f"{snakemake.config['dataset_dir']}dataset/y_test.npy", y_test)
    return 0


def main():
    #split_data("/Thema11/Ipy/results/dataset/full_classification.npy", 1)
    input_array = snakemake.input[0]
    hdf5_output = snakemake.output[0]
    #split_test2("/Thema11/Ipy/results/dataset/full_classification.npy", 1)
    split_test2(input_array, hdf5_output)
    return 0


if __name__ == '__main__':
    with open(snakemake.log[0], "w") as log_file:
        sys.stderr = sys.stdout = log_file
        exitcode = main()
        sys.exit(exitcode)
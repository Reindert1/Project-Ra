import numpy as np
import sys
import gc
from numpy.lib.format import open_memmap
import os


def appender(data_location, outfile, tempfile):
    #total_np = np.load(data_location[0], allow_pickle=True,  mmap_mode='r')
    # total_np = np.load(data_location[0], allow_pickle=True, mmap_mode='c')
    # for i in range(1, len(data_location)):
    #     data = np.load(data_location[i], allow_pickle=True,  mmap_mode='c')
    #     total_np = np.append(total_np, data, axis=1)
    #     del data
    #     gc.collect()
    # total_np = np.unique(total_np, axis=1)
    # np.save(total_np, outfile)

    rows = 0
    cols = 0
    dtype = None
    for data_file in data_location:
        print(data_file)
        data = np.load(data_file, mmap_mode='r')
        rows = data.shape[0]
        cols += data.shape[1]
        dtype = data.dtype

    print(cols)
    print(rows)
    merged = np.memmap(tempfile, dtype=dtype, mode='w+', shape=(rows, cols))
    data_index = 0
    idx = 0
    for data_file in data_location:
        data = np.load(data_file, mmap_mode='c')
        for col in range(data.shape[1]):
            merged[:, data_index] = data[:, col]
            data_index += 1

    disk_array = open_memmap(outfile, mode='w+', dtype=merged.dtype, shape=merged.shape)
    disk_array[:] = merged[:]
    del merged
    gc.collect()
    os.remove(tempfile)

    return 0


def main():
    appender(snakemake.input, snakemake.output[0], snakemake.params["temp_file"])
    return 0


if __name__ == '__main__':
    exitcode = main()
    sys.exit(exitcode)
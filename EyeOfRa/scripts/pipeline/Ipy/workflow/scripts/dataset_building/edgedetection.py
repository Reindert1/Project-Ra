#!usr/bin/env python3

"""
Script to add edge detection feature to dataset for a given image
"""

__author__ = "Skippybal"
__version__ = "0.1"

import cv2 as cv
import numpy as np
import sys


def build_edges(image, save_location):
    image_array = cv.imread(image)
    image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(image_array, (3, 3), 0)

    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_8U, dx=1, dy=1, ksize=5)
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200)
    #print(sobelxy.dtype)
    #print(edges.dtype)

    # cv.imwrite("/commons/Themas/Thema11/Giepmans/work/tmp/sobel.tif", sobelxy)
    #
    # cv.imwrite("/commons/Themas/Thema11/Giepmans/work/tmp/edges.tif", edges)

    full_arr = np.expand_dims(sobelxy.flatten().transpose(), axis=1)
    #print(full_arr.shape)
    full_arr = np.append(full_arr, np.expand_dims(edges.flatten().transpose(), axis=1), axis=1)

    np.save(save_location, full_arr)
    return 0


def main():
    input = snakemake.input[0] #"/commons/Themas/Thema11/Giepmans/Tile_r4-c7_Acquisition Spec 3_452994970.tif" #snakemake.input[0]
    output = snakemake.output[0] #"/commons/Themas/Thema11/Giepmans/" #snakemake.output[0]
    #input = "/commons/Themas/Thema11/Giepmans/work/tmp/larger_data.tif"
    #output = "/commons/Themas/Thema11/Giepmans/work/tmp/larger_data.tif"
    build_edges(input, output)
    return 0


if __name__ == '__main__':
    #with open(snakemake.log[0], "w") as log_file:
        #sys.stderr = sys.stdout = log_file
        exitcode = main()
        sys.exit(exitcode)


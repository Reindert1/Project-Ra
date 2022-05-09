import sys

import cv2

__version__ = 1.0
__author__ = "Reindert1"


def main():
    # image = cv2.imread('Tile_r4-c7_Acquisition Spec 3_452994970.tif')
    # mask = cv2.imread('Sander_r4-c7_nucleus.tif')

    image = sys.argv[1]
    mask = sys.argv[2]

    # Create structuring element, dilate and bitwise-and
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dilate = cv2.dilate(mask, kernel, iterations=3)
    result = cv2.bitwise_and(image, dilate)

    mask[mask == 255] = [255]
    cv2.imshow('dilate_layer.tif', dilate)
    cv2.imshow('result_layer.tif', result)

    return 0


if __name__ == "__main__":
    exitcode = main()
    sys.exit(exitcode)

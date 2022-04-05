# import the necessary packages
from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2


def process(source_path, reference_path):
    # load the source and reference images
    print("[INFO] Loading images")
    source = cv2.imread(source_path)
    reference = cv2.imread(reference_path)

    # determine if we are performing multichannel histogram matching
    # and then perform histogram matching itself
    print("[INFO] Performing matching")
    multi = True if source.shape[-1] > 1 else False
    matched = exposure.match_histograms(source, reference, channel_axis=multi)
    # show the output images
    cv2.imwrite('Tile_r4-c4_Acquisition Spec 3_452994970_matched.tif', matched)
    print("Done")


if __name__ == '__main__':
    source_image = "data/Tile_r4-c4_Acquisition Spec 3_452994970.tif"
    reference_image = "data/Tile_r4-c7_Acquisition_Spec_3_452994970.tif"
    process(source_image, reference_image)

#!usr/bin/env python3

"""
This tool converts images to labeled data.
Non-coloured data is labeled as 0 and coloured data is labeled as 1.
"""

__author__ = "devalk96"
__version__ = "0.1"

import os
import numpy as np
import argparse
import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 268435460


def process(arr: np.array, replacement=255) -> np.array:
    print("Processing image...\nThis might take a while...")
    arr[arr != 0] = replacement
    return arr


def argparser():
    parser = argparse.ArgumentParser(description='Convert tif to labels')
    parser.add_argument("--infile", "-i", help="input filename", required=True)
    parser.add_argument("--outfile", "-o", help="output filename")
    args = parser.parse_args()
    return args


def setup_paths(args) -> [str, str]:
    input = args.infile
    output = args.outfile
    if not output:
        basename = os.path.basename(input)
        output = basename.split(".")[0] + "_corrected." + basename.split(".")[1]
        print(f"No output path provided. Will export to: {output}")
    return input, output


def process_image(input: str) -> np.array:
    image = Image.open(input)
    if image.mode != "L":
        print(f"Converting image loaded in mode: {image.mode} to mode: L")
        image = image.convert("L")
    else:
        print(f"Loaded image in mode: {image.mode}")

    print("Converting to array...")
    arr: np.array = np.array(image)
    processed_array = process(arr, replacement=1)
    return processed_array


def save_image(array, output_path):
    print(f"Saving image to {output_path}")
    Image.fromarray(array).save(output_path)


def main():
    input_path, output_path = setup_paths(argparser())
    result_array = process_image(input_path)
    save_image(result_array, output_path)


if __name__ == '__main__':
    main()

#!usr/bin/env python3

"""
This tool converts overlays multiple images
"""

__author__ = "devalk96"
__version__ = "0.1"

import sys
import argparse
import os
import PIL
from PIL import Image
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = 268435460


def main():
    args = argparser()
    _validate_files(args.overlay + [args.background])
    background_img_path: str = args.background
    overlay_img_path: list[str] = args.overlay

    if args.resize:
        width, height = [int(x) for x in args.resize]
    else:
        width, height = None, None

    img = construct_img(background_img_path, overlay_img_path)

    save_img(img, args.output, width, height)

    print(f"Finished! Saved image at: {args.output}")


def change_alpha(image, background_value=(0, 0, 0)):
    data = np.array(image)

    r1, g1, b1 = background_value
    r2, g2, b2, a2 = 0, 0, 0, 0

    red, green, blue, alpha = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :4][mask] = [r2, g2, b2, a2]

    return Image.fromarray(data)


def construct_img(background, overlay) -> Image.Image:
    alpha_col = (0, 0, 0)
    background = Image.open(background).convert("RGBA")
    for e, image in enumerate(overlay):
        print(f"Overlaying image {e + 1}/{len(overlay)}\t{image}")
        foreground = Image.open(image).convert("RGBA")
        foreground = change_alpha(foreground, background_value=alpha_col)
        background.paste(foreground, (0, 0), foreground)
    return background


def _validate_files(files):
    for file in files:
        if not os.path.exists(file):
            raise FileExistsError(f"File at: {file} does not exist!")


def save_img(img: Image.Image, output_path: str, width=None, height=None):
    if width and height:
        print(f"Resizing output image to : {width}px, {height}px")
        img = img.resize((int(width), int(height)))
    img.save(output_path)


def argparser():
    parser = argparse.ArgumentParser(description='Overlay multiple images')
    parser.add_argument("--background", "-b", help="background filename",
                        required=True)

    parser.add_argument('--overlay', "-o", nargs='+',
                        help="overlay images in order", required=True)

    parser.add_argument("--output", "-O", required=True)

    parser.add_argument("--resize", required=False, help="Resize the image format= 'width height",
                        nargs=2)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    sys.exit(main())

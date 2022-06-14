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
    #args = argparser()

    #_validate_files(args.overlay + [args.background])
    background = snakemake.input["background"]
    overlay = [snakemake.input["overlay"]]

    img = construct_img(background, overlay)

    # if args.resize:
    #     width, height = [int(x) for x in args.resize]
    # else:
    #     width, height = None, None

    output = snakemake.output[0]

    save_img(img, output, None, None)

    print(f"Finished! Saved image at: {output}")
    return 0


def change_alpha(image, background_value=(0, 0, 0)):
    data = np.array(image)

    r1, g1, b1 = background_value
    r2, g2, b2, a2 = 0, 0, 0, 0

    red, green, blue, alpha = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :4][mask] = [r2, g2, b2, a2]

    return Image.fromarray(data)


def label_to_color(image: Image.Image, colornum: int) -> Image.Image:
    colors = [(230, 25, 75, 255),
              (60, 180, 75, 255),
              (255, 225, 25, 255),
              (0, 130, 200, 255),
              (245, 130, 48, 255),
              (145, 30, 180, 255),
              (70, 240, 240, 255),
              (240, 50, 230, 255),
              (210, 245, 60, 255),
              (250, 190, 212, 255),
              (0, 128, 128, 255),
              (220, 190, 255, 255),
              (170, 110, 40, 255),
              (255, 250, 200, 255),
              (128, 0, 0, 255),
              (170, 255, 195, 255),
              (128, 128, 0, 255)]

    data = np.array(image)

    r1, g1, b1 = (1, 1, 1)
    r2, g2, b2, a2 = colors[colornum]
    print(f"\t\t--> Used color is: rgba{colors[colornum]}")

    red, green, blue, alpha = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :4][mask] = [r2, g2, b2, a2]

    return Image.fromarray(data)


def labeled_file_check(image: Image.Image):
    if image.mode != "L":
        print("\t\t--> Image doesn't open in L mode. File is (probably) unlabeled")
        return 0

    colors = image.getcolors()
    if colors[0][1] == 0 and colors[1][1] == 1 and len(colors) == 2:
        print("\t\t--> Image is labeled (contains only two colors)")
        return 1
    else:
        print("\t\t--> Image is not labeled (contains more then two colors): ", image.getcolors())
        return 0


def construct_img(background, overlay) -> Image.Image:
    alpha_col = (0, 0, 0)
    background = Image.open(background).convert("RGBA")
    for e, image in enumerate(overlay):
        print(f"Overlaying image {e + 1}/{len(overlay)}:\t{image}")

        if labeled_file_check(Image.open(image)):
            foreground = label_to_color(Image.open(image).convert("RGBA"), e)
        else:
            foreground = Image.open(image).convert("RGBA")

        foreground = change_alpha(foreground, background_value=alpha_col)
        background.paste(foreground, (0, 0), foreground)
    return background


def _validate_files(files):
    for file in files:
        if not os.path.exists(file):
            raise FileExistsError(f"File at: {file} does not exist!")


def save_img(img: Image.Image, output_path: str, width=None, height=None):
    print(f"Saving final image...")
    img = img.convert("P")

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
    with open(snakemake.log[0], "w") as log_file:
        sys.stderr = sys.stdout = log_file
        sys.exit(main())

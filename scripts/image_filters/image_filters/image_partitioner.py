from PIL import Image
import PIL
import copy
import argparse

PIL.Image.MAX_IMAGE_PIXELS = 268435460


def downscale(image: PIL.Image.Image, scale: float) -> Image.Image:
    """

    :param image: Pillow Image.Image class
    :param scale: Scale: ex 0.1 for 10% or 0.5 for 50%
    :return: Pillow Image.Image class
    """
    new_image = copy.deepcopy(image)
    width, height = new_image.size
    new_image.thumbnail((int(width * scale), int(height * scale)))
    return new_image


def grab_quarter(image: PIL.Image.Image, quarter: int) -> Image.Image:
    """
    Will grab a quarter from an Image and returns as Image
    :param image: Pillow Image.Image class
    :param quarter: Quarter part [1, 2]
                                 [3, 4]
    :return: Pillow Image.Image class
    """

    if quarter not in range(1, 5):
        raise ValueError("Quarter should be between 1 and 4")

    width, height = image.size
    left, top, right, bottom = 0, 0, 0, 0

    if quarter == 1:
        left = 0
        top = 0
        right = int(width * 1 / 2)
        bottom = int(height * 1 / 2)

    elif quarter == 2:
        left = int(width * 1 / 2)
        top = 0
        right = int(width * 1 / 2) * 2
        bottom = int(height * 1 / 2)

    elif quarter == 3:
        left = 0
        top = int(width * 1 / 2)
        right = int(width * 1 / 2)
        bottom = int(width * 1 / 2) * 2

    elif quarter == 4:
        left = int(width * 1 / 2)
        top = int(width * 1 / 2)
        right = int(width * 1 / 2) * 2
        bottom = int(width * 1 / 2) * 2

    return image.crop((left, top, right, bottom))


def _logger(resolution: tuple, downscale: float, quarter: int) -> str:
    names = ["upper-left", "upper-right", "down-left", "down-right"]
    logger = f"Input resolution:\t{resolution}\n" \
             f"After downscaling:\t{tuple([int(x * downscale) for x in resolution]) if downscale != 1.0 else 'No downscaling'}\n" \
             f"Final quarter-size:\t{tuple([int(x * downscale * 0.5) for x in resolution]) if quarter != 0 else 'No quarters'}\n\tQuarter: {names[quarter - 1]}"
    return logger


def _process(image_path: str, outputpath, scale=1, quarter=0):
    image = Image.open(image_path)
    print(_logger(image.size, scale, quarter))
    if scale != 1:
        image = downscale(image, scale)

    if quarter == 0:
        image.save(outputpath)
    else:
        grab_quarter(image, quarter).save(outputpath)


def _argparser():
    parser = argparse.ArgumentParser(description='Easy downscale and quartering of images')
    parser.add_argument("--image", "-i", help="Image filename",
                        required=True)

    parser.add_argument('--downscale', "-d",
                        help="Downscale aspect", required=False, default=1, type=float)

    parser.add_argument('--quarter', "-q",
                        help="Quarter image: Quarter part "
                             "\n[1, 2]"
                             "\n[3, 4]", required=False, default=0, type=int)

    parser.add_argument("--output", "-O", help="Set output path", required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _argparser()
    _process(image_path=args.image, outputpath=args.output,
             scale=args.downscale, quarter=args.quarter)

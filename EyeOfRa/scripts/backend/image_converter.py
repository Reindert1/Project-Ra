from PIL import Image


def convert(image_in, image_out):
    image = Image.open(image_in)
    if image.mode != "RGB":
        image.convert(mode="RGB")
    image.save(image_out)

if __name__ == '__main__':
    infile = "/Users/sanderbouwman/School/Thema11/Themaopdracht/Project-Ra/Project-Ra/EyeOfRa/resources/development/example_images/cropped.tif"

    outfile = "/Users/sanderbouwman/School/Thema11/Themaopdracht/Project-Ra/Project-Ra/EyeOfRa/resources/development/example_images/cropped_converted.png"

    convert(infile, outfile)
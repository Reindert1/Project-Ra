#!/usr/bin/env python

from PIL import Image

im = Image.open('maps_gray.jpg')  # relative path to file

# load the pixel info
pix = im.load()

# get a tuple of the x and y dimensions of the image
width, height = im.size

# open a file to write the pixel data
with open('output_file.csv', 'w+') as f:
    # Potential header
    # f.write('Type: Gray\n')

    # read the details of each pixel and write them to the file
    for x in range(width):
        for y in range(height):
            r = pix[x, y][0]
            g = pix[x, x][1]
            b = pix[x, x][2]
            gray = round((r + g + b) / 3)

            f.write('{0}\n'.format(gray))

            # for coloured values
            f.write('{0}{1}{2}\n'.format(r,g,b))

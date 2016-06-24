# Source masked images are 16-bit RGB compressed TIFFs - relatively large and inefficient
# This routine compresses them to 8-bit greyscale compressed TIFFs that are easier to work with and (probably) w/o
#   any fidelity loss
# Use packbits for compression
#
# Also renames images so time index is consistent.
#   Images are already in alphabetical order, but can be neater.
#   In input set the name "Colony_54_3x3a_Time0070.tif" has a 3x3a part and a Time part. The Time resets when the image
#       get bigger. Use a consistent numbering for time, rearrange order of time and size, for convenience.
# Also rescales intensity. Make images easier to see, but introduces noise artifacts
#
# After looking at all masked images, implement a filter to remove badly exposed regions
#   Mostly early images
#   Need to manually find bad regions of images
#   Alt: "reexpose" those regions by shrinking stuff under the mask
#       Each image has some number of panels: 1st few images have 4 in a 2x2 grid
#       Apply filter to particular panels

import os
from os.path import join as joinpath

import re

from image_loader import image_loader
from save_tiff import save_tiff

from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

# Remember to use raw strings for raw Windows paths (with backslashes)
input_dir = r'C:\Users\xpspectre\Dropbox (MIT)\images_for_Kevin\2_masked_images'
output_dir = r'C:\Users\xpspectre\Downloads\compressed_input_images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

i = 0  # uniform index for image time/numbering

name_format = re.compile(r'^Colony_(\d+)_(\w+)_Time(\d+)$')

for img, name in image_loader(input_dir):
    i += 1
    print('Processing image %d: %s' % (i, name))

    # Rearrange name
    parts = name_format.search(name)
    colony = parts.group(1)  # 0th group is the while match
    size = parts.group(2)
    time = parts.group(3)
    time_pad = len(time)
    new_name = "Colony_{colony}_Time{time:0{width}}_{size}".format(colony=colony, time=i, width=time_pad, size=size)

    # May want to get rid of this step - some images have artifacts that mess this up
    #   Different parts of the image may have different baseline intensities, some with more noise, messing this up
    print("Rescaling intensity...")
    img = img_as_ubyte(rescale_intensity(img))

    print("Converting to greyscale...")
    img = rgb2gray(img)
    save_tiff(joinpath(output_dir, ''.join([new_name, '.tif'])), img)
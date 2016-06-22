# Source masked images are 16-bit RGB compressed TIFFs - relatively large and inefficient
# This routine compresses them to 8-bit greyscale compressed TIFFs that are easier to work with and (probably) w/o
#   any fidelity loss
# Use packbits for compression
# After looking at all masked images, implement a filter to remove badly exposed regions
#   Mostly early images
#   Need to manually find bad regions of images
#   Alt: "reexpose" those regions by shrinking stuff under the mask
#       Each image has some number of panels: 1st few images have 4 in a 2x2 grid
#       Apply filter to particular panels

import os
from os.path import join as joinpath

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

for img, name in image_loader(input_dir):
    print('Processing image %s' % (name,))

    # May want to get rid of this step - some images have artifacts that mess this up
    #   Different parts of the image may have different baseline intensities, some with more noise, messing this up
    print("Rescaling intensity...")
    img = img_as_ubyte(rescale_intensity(img))

    print("Converting to greyscale...")
    img = rgb2gray(img)
    save_tiff(joinpath(output_dir, ''.join([name, '.tif'])), img)
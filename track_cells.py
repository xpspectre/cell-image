# Main script that runs everything

import sys
import os
from os.path import join as joinpath
import argparse

from PIL import Image

import numpy as np

from math import sqrt

import itertools

from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.morphology import closing, skeletonize, reconstruction

import matplotlib.pyplot as plt

import tifffile as tiff

from scipy import ndimage, sparse

# Global parameters

BLUR_RADIUS = 1.2



def analyze_image(filename, output_dir='output', save_figs=False):
    """Get basic information about image"""

    print("Loading...")
    step = 1 # Counter variable for step to make sure all intermediate outputs are in order
    img = plt.imread(filename)
    filebase = os.path.splitext(os.path.basename(filename))[0]

    if save_figs:
        tiff.imsave(joinpath(output_dir, ''.join([filebase, '_', str(step),  '_orig.tif'])), img)

    # Cropping image will make everything after this faster
    #   Make sure this is appropriate for every frame
    print("Cropping...")
    step += 1
    lx, ly, lz = img.shape
    mask = np.zeros((lx, ly))
    img = img[900:lx-100, 900:ly-400, :] # Debugging: replace this later with user-specified bounds

    if save_figs:
        # also save original image with box showing kept region
        tiff.imsave(joinpath(output_dir, ''.join([filebase, '_', str(step),  '_cropped.tif'])), img)

    # Convert RGB image to greyscale and convert back to 8-bit uint
    # Ignore precision loss warning
    print("Converting to greyscale...")
    step += 1
    img = img_as_ubyte(rgb2gray(img))

    if save_figs:
        tiff.imsave(joinpath(output_dir, ''.join([filebase, '_', str(step),  '_greyscale.tif'])), img)

    # Threshold using custom threshold
    #   There's artifacts in the images (should fix the image segmentation output so this doesn't happen) which need to be fixed.
    #   Set a custom, very low threshold instead of trying to automatically calculate one.
    #   Need to maintain connectivity w/o introducing little dots
    # Alternatively use adaptive thresholding
    print('Thresholding...')
    step += 1
    THRESHOLD = 20  # setting > 20 breaks connectivity
    img = img_as_ubyte(img > THRESHOLD)

    if save_figs:
        tiff.imsave(joinpath(output_dir, ''.join([filebase, '_', str(step),  '_thresholded.tif'])), img)

    # Note: this step isn't necessary - just label the regions bigger than a cutoff size
    # Clean up little dots from thresholding
    # print('Removing dots...')
    # step += 1
    # DOTS_REMOVE_RADIUS = 2.5
    # selem = disk(DOTS_REMOVE_RADIUS)
    # img = closing(img, selem)
    #
    # if save_figs:
    #     tiff.imsave(joinpath(output_dir, ''.join([filebase, '_', str(step),  '_dots_removed.tif'])), img)


    # Note/TODO: there may be a built-in function to read images like these and get labeled regions
    #   http://scikit-image.org/docs/dev/auto_examples/plot_label.html
    #   http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
    #   https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html
    # Only keep labels with > some size; count sizes and remove smaller counts
    # Allow parts separated by diagonal to belong to 1 region: s = generate_binary_structure(2,2)
    #   http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_join_segmentations.html#example-segmentation-plot-join-segmentations-py
    #   http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html#example-segmentation-plot-watershed-py
    print("Labeling regions...")
    step += 1

    cleared = img.copy()
    clear_border(cleared)
    label_img = label(img)
    borders = np.logical_xor(img, cleared)
    label_img[borders] = -1
    image_label_overlay = label2rgb(label_img, image=img)

    if save_figs:
        tiff.imsave(joinpath(output_dir, ''.join([filebase, '_', str(step),  '_labeled.tif'])), img)


    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated root length calculator')
    parser.add_argument('-s', '--save-figs', help='Include this flag to save intermediate figs to output directory. Must be first flag.', action='store_true')
    parser.add_argument('-i', '--input', help='Input images directory', required=False, default='images')
    parser.add_argument('-o','--output',  help='Output directory', required=False, default='output')
    parser.add_argument('-t','--temp',  help='Temporary directory for intermediates', required=False, default='temp_')

    args = parser.parse_args()

    # save_figs = args.save_figs
    save_figs = True  # Debugging
    input_dir = args.input
    output_dir = args.output
    temp_dir = args.temp

    if os.path.exists(output_dir):
        print('Warning: Directory %s already exists. Outputs with the same name will overwrite existing files.'%(output_dir,))
    else:
        os.makedirs(output_dir)

    if os.path.exists(temp_dir):
        print('Warning: Temporary directory %s already exists. It''s contents will be overwritten.'%(temp_dir,))
    else:
        os.makedirs(temp_dir)

    files = os.listdir(input_dir)
    for file in files:
        filename = joinpath(input_dir, file)
        print("Processing file %s"%(filename,))

        img = Image.open(filename)
        i = 0
        while True:
            try:
                img.seek(i)
                print('Processing page %d'%(i))
                pagename = joinpath(temp_dir, 'page_%d.tif'%(i,))
                img.save(pagename)
                d = analyze_image(pagename, output_dir=output_dir, save_figs=save_figs)
                i += 1
            except EOFError:
                break

    print('done.')
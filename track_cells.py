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
from skimage.filters import sobel
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential
from skimage.color import label2rgb
from skimage.morphology import closing, skeletonize, reconstruction

import matplotlib.pyplot as plt

import tifffile as tiff

from scipy import ndimage, sparse

# Global parameters


def segment_basic(filename, output_dir='output', temp_dir='temp_', save_figs=False):
    """Get basic information about image"""

    print("Loading...")
    step = 1 # Counter variable for step to make sure all intermediate outputs are in order
    img = plt.imread(filename)
    filebase = os.path.splitext(os.path.basename(filename))[0]

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([filebase, '_', str(step),  '_orig.tif'])), img)

    # Cropping image will make everything after this faster
    #   Make sure this is appropriate for every frame
    print("Cropping...")
    step += 1
    lx, ly, lz = img.shape
    mask = np.zeros((lx, ly))
    img = img[900:lx-100, 900:ly-400, :] # Debugging: replace this later with user-specified bounds

    if save_figs:
        # also save original image with box showing kept region
        tiff.imsave(joinpath(temp_dir, ''.join([filebase, '_', str(step),  '_cropped.tif'])), img)

    # Convert RGB image to greyscale and convert back to 8-bit uint
    # Ignore precision loss warning
    print("Converting to greyscale...")
    step += 1
    img = img_as_ubyte(rgb2gray(img))

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([filebase, '_', str(step),  '_greyscale.tif'])), img)

    # Threshold using custom threshold
    #   There's artifacts in the images (should fix the image segmentation output so this doesn't happen) which need to be fixed.
    #   Set a custom, very low threshold instead of trying to automatically calculate one.
    #   Need to maintain connectivity w/o introducing little dots
    #   Also need to be careful not to join together cells that are close together - such as right after division
    #       This seemed to do OK on the examples, though
    # Alternatively use adaptive thresholding
    print('Thresholding...')
    step += 1
    THRESHOLD = 20  # setting > 20 breaks connectivity
    img = img_as_ubyte(img > THRESHOLD)

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([filebase, '_', str(step),  '_thresholded.tif'])), img)

    # Label regions
    #   http://scikit-image.org/docs/dev/auto_examples/plot_label.html
    #   http://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
    #   https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html
    #   http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_join_segmentations.html#example-segmentation-plot-join-segmentations-py
    #   http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html#example-segmentation-plot-watershed-py
    # http://cmm.ensmp.fr/~beucher/wtshed.html
    # Note/TODO: Will probably need watershed and more fancy methods for the real segmentation
    print("Labeling regions...")
    step += 1

    label_img, n_labels = label(img, connectivity=2, return_num=True)
    regions = regionprops(label_img)

    # Keep the labeled regions bigger than some cutoff area
    #   The larger regions are cells, the smaller regions are noise
    REGION_AREA_CUTOFF = 10  # px
    kept_regions = []
    kept_labels = []
    for region in regions:
        if region.area > REGION_AREA_CUTOFF:
            kept_regions.append(region)
            kept_labels.append(region.label)

    label_img_cleaned = label_img
    label_img_cleaned[np.in1d(label_img, kept_labels, invert=True).reshape(label_img.shape)] = 0  # Make a mask of all positions not in kept_labels
    label_img_cleaned, _, _ = relabel_sequential(label_img_cleaned)

    # Get image with only the cells
    img = img_as_ubyte(label_img_cleaned > 0)

    # Display regions with unique colors
    label_img_overlay = label2rgb(label_img_cleaned, image=img)

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([filebase, '_', str(step), '_labeled.tif'])),
                    img_as_ubyte(img))
        tiff.imsave(joinpath(temp_dir, ''.join([filebase, '_', str(step), '_labeled_overlay.tif'])),
                    img_as_ubyte(label_img_overlay))


    # TODO:Output stats for regions
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
                shortfile = file.split('.')[0]  # remove file extension(s)
                pagename = joinpath(temp_dir, shortfile + '_page_%d.tif'%(i,))
                img.save(pagename)
                d = segment_basic(pagename, output_dir=output_dir, temp_dir=temp_dir, save_figs=save_figs)
                i += 1
            except EOFError:
                break

    print('done.')
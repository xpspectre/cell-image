# Test out segmenting a a harder, raw image
# Observations
#   The original images' intensities are scaled to a very small part of the overall space of uint16's. Start off by
#       rescaling to the full length to make stuff visible
#   The images are 3x3 stitched together. At the seams, differences in contrast are visible.
#   There's a bunch of artifacts (I think) that look like circles/rings. They look like they're smaller than the cells,
#       and definitely have different shape (they look rounder and have big holes in them). May be able to make a simple
#       classifier that rejects them. However, some of the cells are also pretty round.
#   Warning: these images can be huge, so be careful with saving intermediates
#   Note: Getting the long, thin appendages of cells isn't that important for centroid/position and area calculations
#       (which are used for those cell tracking plots) but they are important for calculating stats on the overall
#       shape (which are important for any machine learning technique for classifying cells by shape).
#   More sophisticated ways of doing this segmentation would be to look at the candidate regions and running them
#       through a classifier trained on previous known cells
#   An even more sophisticated way would be on-line with cell tracking - saying that a candidate region is more likely a
#       cell if it's in a position close to a cell in the previous frame
#       - This is particularly important when considering dividing cells - when all images are processed independently,
#         slight differences in contrast and other issues from fine-tuning thresholds can lead to 2 cells being read as
#         the same or different cells in successive images. This would confuse a naive cell tracker being fed with
#         regions that appeared to joint and separate in successive frames. On-line cell tracking can enforce the fact
#         that cells only divide and provide some kind of mechanism to break up connected regions.
#   I think no matter how good the fully automated approach gets, it won't be as good as manually marking cells.
#       But it will be a lot faster/easier. Probably worth it.

import os
from os.path import join as joinpath

from PIL import Image

import numpy as np

from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.morphology import remove_small_objects, watershed
from skimage.filters import threshold_otsu, sobel, scharr

import matplotlib.pyplot as plt

import tifffile as tiff

import re
import json
from NumpyJSONEncoder import NumpyJSONEncoder

from image_loader import image_loader


def segment_test(img, name, output_dir='output', temp_dir='temp', save_figs=False):
    """Test segmentation on harder images from earlier in the pipeline."""

    step = 1 # Counter variable for step to make sure all intermediate outputs are in order

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step),  '_orig.tif'])), img)

    # Everything appears as black in the initial image - it was just rescaled to
    print("Rescaling intensity...")
    step += 1
    img = rescale_intensity(img)

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step),  '_rescaled.tif'])), img)

    # Convert RGB image to greyscale and convert back to 8-bit uint
    # Ignore precision loss warning
    print("Converting to greyscale...")
    step += 1
    img = img_as_ubyte(rgb2gray(img))

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step),  '_greyscale.tif'])), img)

    # # Basic thresholding for segmentation
    # #   This probably doesn't work well enough because of:
    # #       - Faint parts of cells get removed - big problem
    # #       - Artifacts remain - this isn't that big a problem - they can be removed later
    # print('Basic thresholding...')
    # step += 1
    # # THRESHOLD = 20  # setting > 20 breaks connectivity
    # threshold = threshold_otsu(img)
    # img = img_as_ubyte(img >= threshold)
    #
    # if save_figs:
    #     tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step),  '_thresholded.tif'])), img)

    # More sophisticated watershed segmentation
    # Basic idea:
    #   - Find markers/centers/pixels that are definitely in cells. These look like the brightest ones but some fine-
    #       tuning may be needed based on how bright the brightest artifacts are
    #   - Find the edges of the cells. There are 2 ways to proceed:
    #       1. Use usual edge finding algorithm like Sobel.
    #       2. Use thresholding technique (either a manually set threshold or Otsu or similar) to determine region that
    #           may include a cell and apply Sobel to that. It's sharper but requires tuning.
    #   - Watershed starts from the markers and fills to the edges. This basically gives you a 2-step thresholding
    #       technique. It seems like it takes care of artifacts well. Some brighter artifacts may still remain, though.
    print('Region-based segmenting')
    step += 1

    # Get centers
    #   Ideally, white regions are inside cells and no white regions are inside artifacts
    #   This should work well because the cells are "brighter" than the artifacts
    print('Getting makers')
    markers = np.zeros_like(img)
    MARKER_LO_THRESHOLD = 65  # sensitive/fine-tuned
    # MARKER_LO_THRESHOLD = threshold_otsu(img)  # this may work but general sets the threshold too high
    MARKER_HI_THRESHOLD = 150  # sensitive/fine-tuned
    markers[img < MARKER_LO_THRESHOLD] = 1
    markers[img > MARKER_HI_THRESHOLD] = 2

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step), '_markers.tif'])), img_as_ubyte(rescale_intensity(markers)))

    # # Get edges
    # print('Getting edges using gradient methods')
    # elevation_map = sobel(img)
    # # elevation_map = scharr(img)
    # # elevation_map[elevation_map > 200] = 200
    # # elevation_map = rescale_intensity(elevation_map)
    # if save_figs:
    #     tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step), '_elevation_map.tif'])), img_as_ubyte(elevation_map))

    # Much more aggressive than direct gradient finding on image
    print('Getting candidate regions (regions that may be cells)')
    candidate_regions = np.zeros_like(img)
    candidate_regions[img > MARKER_LO_THRESHOLD] = 1
    candidate_regions = rescale_intensity(candidate_regions)
    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step), '_candidate_regions.tif'])), rescale_intensity(candidate_regions))

    print('Getting edges of candidate regions')
    elevation_map = sobel(candidate_regions)
    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step), '_elevation_map.tif'])), img_as_ubyte(elevation_map))

    # Do watershed transform
    #   Warning: expensive
    print('Doing watershed')
    img = rescale_intensity(watershed(elevation_map, markers))

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step), '_watershed.tif'])), img)

    # Split close cells
    #   This step is sensitive/fine-tuned
    #   A bunch of long, thin cells that may or may not be linked is hard to handle
    #   One requirement is that when cells split, they can't merge afterwards. Requires online cell tracking.
    # Another watershedding step
    #   Make markers by distances from edge - identifies centers of blobs/cells - this works for round cells (and can
    #       split touching cells) but not as well for elongated cells, where it's hard to tell the split.
    #
    # print('Splitting cells')
    #
    # if save_figs:
    #     tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step), '_split.tif'])), img)

    # Remove small artifacts
    print("Removing small objects...")
    step += 1
    OBJECT_SIZE_THRESHOLD = 1000
    img = img_as_ubyte(remove_small_objects(img > 100, min_size=OBJECT_SIZE_THRESHOLD))  # run remove_small_objects on a boolean matrix

    if save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step),  '_small_objects_removed.tif'])), img)

    # DEBUG: still output final result when not saving figs
    if not save_figs:
        tiff.imsave(joinpath(temp_dir, ''.join([name, '_', str(step), '_segmented.tif'])), img)

    # Label regions

    return 0

if __name__ == "__main__":
    save_figs = False  # True when developing the pipeline
    input_dir = 'images'
    output_dir = 'output'
    temp_dir = 'temp'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # (Re-) segment images into cells
    segmented_results = []
    for img, name in image_loader(input_dir):
        print('Processing image %s' % (name,))
        cells = segment_test(img, name, output_dir=output_dir, temp_dir=temp_dir, save_figs=save_figs)

        tokens = re.findall('.+Time(\d+)', name)
        time = int(tokens[0])  # hopefully this works...

        stats = {'time': time, 'cells': cells}

        segmented_results.append(stats)

    # Output results
    print('Outputting segmented results in JSON format')
    segmented_results_file = joinpath(output_dir, 'segmented_results.txt')
    with open(segmented_results_file, 'w') as f:
        json.dump(segmented_results, f, cls=NumpyJSONEncoder, indent=4, sort_keys=True)

    print('done.')
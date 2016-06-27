# Main script that sequentially finds and tracks cells

import os
from os.path import join as joinpath
import re
import csv
import ast

from image_loader import image_loader
from save_tiff import save_tiff

from skimage.draw import polygon

# Constants
# Numpy stores matrices as a col vector of row vectors. Images indexed as (y,x)
IMAGE_HEIGHT = 5070
IMAGE_WIDTH = 6720
PANEL_HEIGHT = int(IMAGE_HEIGHT/5)
PANEL_WIDTH = int(IMAGE_WIDTH/5)
CENTER = (IMAGE_HEIGHT/2, IMAGE_WIDTH/2)

# Default locations
input_dir = 'images'
output_dir = 'output'
save_figs = True

# Load instructions for preprocessing panels
# Noisy panels
#   The 3rd col of this csv file is a tuple of tuples
noisy_panels_filename = 'extra_inputs/noisy_panels.csv'
noisy_panels = {}
with open(noisy_panels_filename, newline='') as noisy_panels_file:
    reader = csv.reader(noisy_panels_file, delimiter=',', quotechar='"')
    next(reader)  # ignore 1st line with column headers
    for row in reader:
        # Hopefully these are in the right format...
        start = int(row[0])
        end = int(row[1])
        panels = ast.literal_eval(row[2])
        for i in range(start, end+1):
            noisy_panels[i] = panels

# Irregularly shaped panels
#   Assume 1 cutout region/image
#   Points specify polygon in original (uncropped) image
cutout_filename = 'extra_inputs/cutout_regions.csv'
cutout_regions = {}
with open(cutout_filename, newline='') as cutout_file:
    reader = csv.reader(cutout_file, delimiter=',', quotechar='"')
    next(reader)  # ignore 1st line with column headers
    for row in reader:
        start = int(row[0])
        end = int(row[1])
        points = ast.literal_eval(row[2])
        for i in range(start, end+1):
            cutout_regions[i] = points

# Load approximate cell starting positions (and count)
#   Positions are for original (uncropped) image
starting_filename = 'extra_inputs/starting_cells.csv'
starting_positions = []
with open(starting_filename, newline='') as starting_file:
    reader = csv.reader(starting_file, delimiter=',', quotechar='"')
    next(reader)  # ignore 1st line with column headers
    for row in reader:
        for pos in row:
            starting_positions.append(ast.literal_eval(pos))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

name_format = re.compile(r'^Colony_(\d+)_Time(\d+)_(\w+)$')
size_format = re.compile(r'^(\d)x\da$')

for img, name in image_loader(input_dir):
    print('Processing image %s' % (name,))

    step = 0  # Counter to keep track of image step
    if save_figs:
        save_tiff(joinpath(output_dir, ''.join([name, '_', str(step), '_original.tif'])), img)

    parts = name_format.search(name)
    colony = int(parts.group(1))  # not used
    time = int(parts.group(2))
    size_parts = size_format.search(parts.group(3))
    size = int(size_parts.group(1))

    print('Preprocessing sections')

    # Make sure image is the expected dimensions
    #   Needed to make sure image sections are correct
    if not img.shape == (IMAGE_HEIGHT,IMAGE_WIDTH):
        print('Image dimensions don''t match expected. Got %d x %d' % (img.shape[0], img.shape[1]))

    # Cutout polygon for big artifacts
    if time in cutout_regions:
        points = cutout_regions[time]
        ys = [point[0] for point in points]
        xs = [point[1] for point in points]
        rr, cc = polygon(ys, xs)
        img[rr, cc] = 0

    # Crop img to just the panels actually taken by the camera
    #   offset when tracking stuff is nw corner
    #   Smaller/faster to work with
    #   Also makes handling cells that leave off the edge (if there are any) appropriately
    nw = (int(CENTER[0] - PANEL_HEIGHT*size/2), int(CENTER[1] - PANEL_WIDTH*size/2))
    se = (int(CENTER[0] + PANEL_HEIGHT*size/2), int(CENTER[1] + PANEL_WIDTH*size/2))
    img = img[nw[0]:se[0], nw[1]:se[1]]

    step += 1
    if save_figs:
        save_tiff(joinpath(output_dir, ''.join([name, '_', str(step), '_cropped.tif'])), img)

    # Remove noise artifacts from certain panels when stitching them together
    #   Noisy panels don't have cells - the camera normalizes the noise intensity when there's no real cells (or debris)
    #   to see
    if time in noisy_panels:
        removed_panels = noisy_panels[time]
        for panel in removed_panels:
            (panel_y, panel_x) = panel
            panel_nw = ((panel_y - 1) * PANEL_HEIGHT, (panel_x - 1) * PANEL_WIDTH)
            panel_se = (panel_y * PANEL_HEIGHT, panel_x * PANEL_WIDTH)
            img[panel_nw[0]:panel_se[0], panel_nw[1]:panel_se[1]] = 0

    step += 1
    if save_figs:
        save_tiff(joinpath(output_dir, ''.join([name, '_', str(step), '_noisy_panels_blanked.tif'])), img)


    1
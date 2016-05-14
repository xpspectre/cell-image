#!/usr/bin/env python3
# Main script for running entire pipeline
import argparse
import os
from os.path import join as joinpath
import re

import json
from NumpyJSONEncoder import NumpyJSONEncoder

from segment_cells import segment_basic
from track_cells import track_cells_basic
from image_loader import image_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated root length calculator')
    parser.add_argument('-s', '--save-figs', help='Include this flag to save intermediate figs to output directory. Must be first flag.', action='store_true')
    parser.add_argument('-i', '--input', help='Input images directory', required=False, default='images')
    parser.add_argument('-o','--output',  help='Output directory', required=False, default='output')
    parser.add_argument('-t','--temp',  help='Temporary directory for intermediates', required=False, default='temp')

    args = parser.parse_args()

    save_figs = args.save_figs
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

    # (Re-) segment images into cells
    segmented_results = []
    for img, name in image_loader(input_dir):
        print('Processing image %s' % (name,))
        cells = segment_basic(img, name, output_dir=output_dir, temp_dir=temp_dir, save_figs=save_figs)

        tokens = re.findall('.+Time(\d+)', name)
        time = int(tokens[0])  # hopefully this works...

        stats = {'time': time, 'cells': cells}

        segmented_results.append(stats)

    # Output segmented results
    print('Outputting segmented results in JSON format')
    segmented_results_file = joinpath(output_dir, 'segmented_results.txt')
    with open(segmented_results_file, 'w') as f:
        json.dump(segmented_results, f, cls=NumpyJSONEncoder, indent=4, sort_keys=True)

    # Track cells
    print('Tracking cells')
    tracked_results = track_cells_basic(segmented_results)

    # Output tracked results
    print('Outputting tracked results in JSON format')
    tracked_results_file = joinpath(output_dir, 'tracked_results.txt')
    with open(tracked_results_file, 'w') as f:
        json.dump(tracked_results, f, cls=NumpyJSONEncoder, indent=4, sort_keys=True)

    print('done.')

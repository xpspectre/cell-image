import os
from os.path import join as joinpath
from os.path import isfile
import tifffile as tiff


def image_loader(input_dir):
    """Generator that returns TIFF images in input_dir. Treats pages of multi-page TIFFs as individual images.
    TODO: also return image metadata.

    Usage:
        for img, name in image_loader(input_dir):
            <do something with img and name>
    """

    files = [f for f in os.listdir(input_dir) if isfile(joinpath(input_dir, f))]  # get the files, excluding directories
    for file in files:
        filename = joinpath(input_dir, file)
        with tiff.TiffFile(filename) as tif:
            i = 0
            for page in tif:
                i += 1

                img = page.asarray()
                if len(tif.pages) == 1: # Handle usual case of single-page TIFF w/o page number in name
                    name = file.split('.')[0]
                else:
                    name = file.split('.')[0] + '_page_%d'%(i,)

                yield (img, name)
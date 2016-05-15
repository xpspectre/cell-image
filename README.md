# cell-image

Tools for segmenting and tracking cells and analyzing them.

Initialized from root-image project.

## Setup Instructions:

### Windows:
   1. Download and install Anaconda (download link: http://repo.continuum.io/archive/Anaconda2-2.5.0-Windows-x86_64.exe)
   2. Enable conda-forge and install tifffile. See instructions at https://github.com/conda-forge/tifffile-feedstock
   3. (This can also be done in a virtualenv)

### Linux:
   1. Make a virtualenv in an env/ subdir of the main directory and activate it
   2. Install packages: pip install -r requirements.txt (this file was generated from pip freeze > requirements.txt)
       The exact package versions probably aren't necessary - it works with a bunch - but you may need a new-ish version to support all the TIFF features.

## Usage:

The main script is ``run_pipeline.py``. See options by running ``run_pipeline.py -h``.

Helper scripts/libraries for pipeline steps are ``segment_cells.py`` and ``track_cells.py``.

## Notes:
   - Received images for cell tracking have been slightly postprocessed after segmentation and have some artifacts (are those from jpg?). Make sure the whole integrated workflow doesn't do this.
   - The CellProfiler sample pipeline output final images as JPEG. Don't want to do that - make sure you use lossless compression for everything.
   - Load tiffs source: https://stackoverflow.com/questions/7569553/working-with-tiffs-import-export-in-python-using-numpy
   - Warning: intermediate files may be very big uncompressed TIFFs
   - Cell tracking is much faster than CellProfiler (at least for CellProfiler on Windows). Aside from slightly cleaning up the already segmented images (that got saved as JPEGs), it's instant.

"""
Handles the ETL for Ciders-Sinai dataset.
"""

import cv2
import numpy as np
import os
path = os.path.expandvars("$TMPDIR/TIFF color normalized sequential filenames/test100.tif")

patch_size = 29
stride = 10

img = cv2.imread(path)

def patch_iter(img, patch_size, stride):
    '''
    Takes an image represented by a numpy array of dimensions [l, w,
    channel], and creates an iterator over all patches in the imagine
    defined by the `patch_size` and the `stride` length.
    '''
    # TODO unit test
    for x in range((img.shape[0]-patch_size+1)/stride):
        for y in range((img.shape[1]-patch_size+1)/stride):
            patch = img[x*stride:x*stride+patch_size, y*stride:y*stride+patch_size, :]
            yield patch

patches = patch_iter(img, patch_size, stride)



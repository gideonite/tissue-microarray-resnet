"""
Handles the ETL for Ciders-Sinai dataset.
"""

from collections import namedtuple
import cv2
import itertools
import numpy as np
import random
import scipy.io as sio
import os

random.seed(1337)

patch_size = 29
stride = 10

def patch_iter(imgfilename, patch_size, stride):
    '''
    Takes an image represented by a numpy array of dimensions [l, w,
    channel], and creates an iterator over all patches in the imagine
    defined by the `patch_size` and the `stride` length.
    '''
    # TODO unit test
    img = cv2.imread(imgfilename)
    for x in range((img.shape[0]-patch_size+1)/stride):
        for y in range((img.shape[1]-patch_size+1)/stride):
            patch = img[x*stride:x*stride+patch_size, y*stride:y*stride+patch_size]
            yield patch

def center_of_patch(patch):
    length = patch.shape[0]
    width = patch.shape[1]
    return patch[length/2,width/2]

def patch_center_iter(imgfilename, patch_size, stride):
    img = cv2.imread(imgfilename)
    for patch in patch_iter(img, patch_size, stride):
        yield patch[patch_size/2, patch_size/2]

xdata = []
ydata = []
img_filename = "$TMPDIR/TIFF color normalized sequential filenames/test%d.tif"
label_filename = "$TMPDIR/Color annotation sequential filenames/test%d_Annotated.tif"
file_nums = list(range(1,256))
random.shuffle(file_nums)
for file_num in file_nums:
    xdata.append(patch_iter(os.path.expandvars(img_filename %(file_num)), patch_size, stride))
    ydata.append(patch_center_iter(os.path.expandvars(label_filename %(file_num)), patch_size, stride))

DataSet = namedtuple("DataSet", ["xdata", "ydata"])
cedars_sinai_data = DataSet(itertools.chain(*xdata), itertools.chain(*ydata))

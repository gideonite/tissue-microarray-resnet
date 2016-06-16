"""
Handles the ETL for Ciders-Sinai dataset.
"""

import cv2
import itertools
import numpy as np
import random
import scipy.io as sio
import os

random.seed(1337) # TODO

patch_size = 29
stride = 10
img_filename = "/mnt/data/TIFF color normalized sequential filenames/test%d.tif"
label_filename = "/mnt/data/ATmask sequential filenames/test%d_Mask.mat"

def _patches(imgfilename, patch_size, stride):
    '''
    Takes an image represented by a numpy array of dimensions [l, w,
    channel], and creates an iterator over all patches in the imagine
    defined by the `patch_size` and the `stride` length.
    '''
    img = cv2.imread(imgfilename).astype('float32')
    ret = []
    for x in range((img.shape[0]-patch_size+1)/stride):
        for y in range((img.shape[1]-patch_size+1)/stride):
            patch = img[x*stride:x*stride+patch_size, y*stride:y*stride+patch_size]
            ret.append(patch.flatten())

    return ret

def _patch_labels(matfilename, patch_size, stride):
    '''
    Takes a patch of pixel-wise labels and extracts the representative
    label, namely the center of the patch.
    '''
    img = sio.loadmat(matfilename)['ATmask']
    ret = []
    for x in range((img.shape[0]-patch_size+1)/stride):
        for y in range((img.shape[1]-patch_size+1)/stride):
            patch = img[x*stride:x*stride+patch_size, y*stride:y*stride+patch_size]
            ret.append(patch[patch_size/2, patch_size/2] - 1)
    return ret

file_nums = list(range(1,225))
random.shuffle(file_nums)
def dataset(num_images=len(file_nums), train_test_split=0.8):
    print('reading ' + str(num_images) + ' images')
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []

    for file_num in file_nums[:int(train_test_split * num_images)]:
        patches = _patches(img_filename %(file_num), patch_size, stride)
        labels = _patch_labels(label_filename %(file_num), patch_size, stride)

        assert len(patches) == len(labels)

        for i in xrange(len(patches)):
            xtrain.append(patches[i])
            ytrain.append(labels[i])

    for file_num in file_nums[int(train_test_split * num_images):num_images]:
        patches = _patches(img_filename %(file_num), patch_size, stride)
        labels = _patch_labels(label_filename %(file_num), patch_size, stride)

        assert len(patches) == len(labels)

        for i in xrange(len(patches)):
            xtest.append(patches[i])
            ytest.append(labels[i])

    return np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest)

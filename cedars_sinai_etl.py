"""
Handles the ETL for Ciders-Sinai dataset.
"""

import cv2
import itertools
import numpy as np
import random
import scipy.io as sio
import os
import tensorflow as tf

random.seed(1337) # TODO

img_filename = "/mnt/data/TIFF color normalized sequential filenames/test%d.tif"
label_filename = "/mnt/data/ATmask sequential filenames/test%d_Mask.mat"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('patch_size', 64, 'Size of square patches to extract from images')
flags.DEFINE_integer('stride', 32, 'Stride between patches')

def _patches(imgfilename, patch_size=FLAGS.patch_size, stride=FLAGS.stride):
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

def _patch_labels(matfilename, patch_size=FLAGS.patch_size, stride=FLAGS.stride):
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

_file_nums = list(range(1,225))
random.shuffle(_file_nums)
def dataset(num_images=len(_file_nums)):
    print('reading ' + str(num_images) + ' images')
    print('patch_size=' + str(FLAGS.patch_size) + ' stride=' + str(FLAGS.stride))

    xdata, ydata = [], []
    for file_num in _file_nums[:num_images]:
        patches = _patches(img_filename %(file_num))
        labels = _patch_labels(label_filename %(file_num))
        assert len(patches) == len(labels)

        for i in xrange(len(patches)):
            xdata.append(patches[i])
            ydata.append(labels[i])

    return np.array(xdata), np.array(ydata)

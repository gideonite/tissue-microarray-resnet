"""
Handles the ETL for Ciders-Sinai dataset.
"""

import cv2
import itertools
import numpy as np
import scipy.io as sio
import os
import os.path
import tensorflow as tf

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
    img = cv2.imread(imgfilename)
    if img is None:
        raise Exception("File not found: " + '"' + imgfilename + '"')
    img = img.astype('float32')
    ret = []
    for x in range(int((img.shape[0]-patch_size+1)/stride)):
        for y in range(int((img.shape[1]-patch_size+1)/stride)):
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
    for x in range(int((img.shape[0]-patch_size+1)/stride)):
        for y in range(int((img.shape[1]-patch_size+1)/stride)):
            patch = img[x*stride:x*stride+patch_size, y*stride:y*stride+patch_size]
            label_value = patch[patch_size/2, patch_size/2]
            label_value -= 1 # need to start from 0.
            ret.append(label_value)
    return ret

def dataset(path='.', split=0.8, random_seed=1337):
    '''
    Returns shuffled training data of paired patches with
    labels. Returns a tuple, (X, y).
    '''

    # if we've done this before, just reuse it.
    if os.path.exists(path + '/xtrain.npy') and \
       os.path.exists(path + '/xtest.npy') and \
       os.path.exists(path + '/ytrain.npy') and \
       os.path.exists(path + '/ytest.npy'):
        
        xtrain = np.load(path + '/xtrain.npy')
        xtest = np.load(path + '/xtest.npy')
        ytrain = np.load(path + '/ytrain.npy')
        ytest = np.load(path + '/ytest.npy')
        return xtrain, xtest, ytrain, ytest

    print('creating new data for training and testing')
    if random_seed:
        np.random.seed(random_seed)

    xdata, ydata = [], []
    num_images=224
    for file_num in range(1, num_images+1):
        patches = _patches(img_filename %(file_num))
        labels = _patch_labels(label_filename %(file_num))
        assert len(patches) == len(labels)

        for i in xrange(len(patches)):
            xdata.append(patches[i])
            ydata.append(labels[i])

    idx = np.array(list(range(len(xdata))))
    np.random.shuffle(idx)
    xdata = np.array(xdata)[idx]
    ydata = np.array(ydata)[idx]

    assert len(xdata) == len(ydata)
    num_examples = len(xdata)

    # set aside a "do not touch set"
    ten_percent = int(0.1*num_examples)
    xdonottouch = xdata[:ten_percent]
    ydonottouch = ydata[:ten_percent]
    np.save(path + '/xdonottouch.npy', xdonottouch)
    np.save(path + '/ydonottouch.npy', ydonottouch)
    del xdonottouch, ydonottouch

    # now pretend that we never had the "do not touch set"
    xdata = xdata[ten_percent:]
    ydata = ydata[ten_percent:]
    num_examples = len(xdata)

    pivot = int(split*num_examples)
    xtrain = xdata[:pivot, :]
    xtest = xdata[pivot:, :]
    ytrain = ydata[:pivot]
    ytest = ydata[pivot:]

    np.save(path + '/xtrain.npy', xtrain)
    np.save(path + '/xtest.npy', xtest)
    np.save(path + '/ytrain.npy', ytrain)
    np.save(path + '/ytest.npy', ytest)

    return xtrain, xtest, ytrain, ytest

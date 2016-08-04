import cv2
import itertools
import numpy as np
import os
import os.path
import random
import tensorflow as tf
import scipy.io as sio

# img_filename = "/mnt/data/TIFF color normalized sequential filenames/test%s.tif"
# label_filename = "/mnt/data/ATmask sequential filenames/test%s_Mask.mat"

def read_list_of_numbers_or_fail(filename):
    with open(filename) as f:
        return [int(l.strip()) for l in f.readlines()]

try:
    trainset = read_list_of_numbers_or_fail("/mnt/data/train.txt")
    valset = read_list_of_numbers_or_fail("/mnt/data/validation.txt")
except IOError, e:
    raise IOError("Must define training and validation sets as files with a list of numbers corresponding to images.\n" + str(e))

def imread(example_num):
    '''
    For some reason cv2 doesn't throw an error when it doesn't find
    the file. I want to.
    '''
    filename = "/mnt/data/TIFF color normalized sequential filenames/test%d.tif" % example_num
    img = cv2.imread(filename)

    if img is None:
        raise Exception("File not found: " + '"' + filename + '"')

    return img.astype('float32')

def labelread(example_num):
    filename = "/mnt/data/ATmask sequential filenames/test%d_Mask.mat" % example_num
    labels = sio.loadmat(filename)['ATmask']
    return labels

def center_pixel(patch):
    '''
    Takes a patch of pixel-wise labels and extracts the representative
    label, namely the center of the patch.
    '''
    length, height = patch.shape[:2]
    return np.array([patch[length/2, height/2]-1]) # labels are 0-indexed.

def dataset(patch_size, stride, batch_size, label_f):
    train_data = [(imread(ex_num), labelread(ex_num)) for ex_num in trainset]
    # setup possible patch coordinates
    for i in range(batch_size):
        x,y = random.choice(train_data)

        # select randomly from the patch coordinates and index into the images
        # append to batch
        # yield xbatch, ybatch

dataset(64, 15, 64, center_pixel)

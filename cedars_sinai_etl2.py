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

def _patches(img, patch_size, stride):
    assert 2 <= len(img.shape) <= 3
    num_xpatches = int((img.shape[0]-patch_size+1) / stride)
    num_ypatches = int((img.shape[1]-patch_size+1) / stride)

    #blah
    ret = []
    for x in range(0, img.shape[0]-patch_size+1, stride):
        for y in range(0, img.shape[1]-patch_size+1, stride):
            ret.append(img[x : x+patch_size, y : y+patch_size])
    return ret

def _load(patch_size, stride, batch_size, label_f):
    train_data = [(imread(ex_num), labelread(ex_num)) for ex_num in trainset]

    xs, ys = zip(*[(_patches(x, patch_size, stride), _patches(y, patch_size, stride)) for x,y in train_data])
    xs = [x for sublist in xs for x in sublist]
    ys = [y for sublist in ys for y in sublist]

    num_examples = len(xs)
    assert len(xs) == len(ys)

    return xs, ys

def dataset(patch_size, stride, batch_size, label_f):
    xs, ys = _load(patch_size, stride, batch_size, label_f)
    num_examples = len(xs)
    def iter():
        while True:
            xbatch, ybatch = [], []
            for _ in range(batch_size):
                i = random.randrange(num_examples)
                num_rot = random.choice([0,1,2,3])
                x = np.rot90(xs[i], num_rot)
                xbatch.append(x)
                ybatch.append(ys[i])

            yield np.array(xbatch), np.array([label_f(y)[0] for y in ybatch])

    return num_examples, iter

def minidata(patch_size, stride, batch_size, label_f):
    xs, ys = _load(patch_size, stride, batch_size, label_f)
    xpatch = xs[42]
    ypatch = ys[42]

    def iter():
        while True:
            yield np.array([xpatch for i in range(batch_size)]), np.squeeze(np.array([label_f(ypatch) for i in range(batch_size)]))

    return 1, iter

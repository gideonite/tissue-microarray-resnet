"""
Handles the ETL for Ciders-Sinai dataset.
"""

import cv2
import itertools
import numpy as np
import os
import os.path
import random
import tensorflow as tf
import scipy.io as sio

img_filename = "/mnt/data/TIFF color normalized sequential filenames/test%d.tif"
label_filename = "/mnt/data/ATmask sequential filenames/test%d_Mask.mat"

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('frac_data', 1.0, 'Fraction of training data to use')
flags.DEFINE_integer('patch_size', 64, 'Size of square patches to extract from images')
flags.DEFINE_integer('stride', 32, 'Stride between patches')

def imread(filename):
    '''
    For some reason cv2 doesn't throw an error when it doesn't find
    the file. I want to.
    '''
    img = cv2.imread(filename)

    if img is None:
        raise Exception("File not found: " + '"' + filename + '"')

    return img.astype('float32')


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

def center_pixel(patch):
    '''
    Takes a patch of pixel-wise labels and extracts the representative
    label, namely the center of the patch.
    '''
    length, height = patch.shape[:2]
    return np.array([patch[length/2, height/2]])

def collapse_classes(classes):
    classes[classes == 1] = 0
    classes[classes == 3] = 0

    classes[classes == 2] = 1
    classes[classes == 4] = 1

    return classes

def _load_data():
    xdata, ydata = [], []
    num_images=224
    for file_num in range(1, num_images+1):
        img = imread(img_filename % file_num)
        labels = sio.loadmat(label_filename % file_num)['ATmask']

        xdata.append(img)
        ydata.append(labels)

    return xdata, ydata

def _shuffle_xy_pairs(xdata, ydata):
    num_examples = len(xdata)
    assert num_examples == len(ydata)
    
    idx = np.array(list(range(num_examples)))
    np.random.shuffle(idx)
    xdata = np.array(xdata)[idx]
    ydata = np.array(ydata)[idx]
    return xdata, ydata

def _maybe_create_dataset(path, patch_size, stride, split=0.8):
    train_filename = path + "/%strain_patchsize" + str(patch_size) + '_stride' + str(stride) + '.npy'
    val_filename = path + "/%sval_patchsize" + str(patch_size) + '_stride' + str(stride) + '.npy'
    test_filename = path + "/%stest_patchsize" + str(patch_size) + '_stride' + str(stride) + '.npy'

    filenames = [train_filename, val_filename, test_filename]

    existing_files = []
    for f in filenames:
        for x in ['x','y']:
            existing_files.append(os.path.exists(f % x))
    existing_files = set(existing_files)
    assert(len(existing_files)) == 1, "Somehow not all slices of dataset were saved"

    dataset_exists = not list(existing_files)[0]
    if dataset_exists:
        print('creating new data for training and testing')

        xdata, ydata = _load_data()

        xpatches = []
        for img in [_patches(x, patch_size, stride) for x in xdata]:
            for patch in img:
                xpatches.append(patch)

        ypatches = []
        for img in [_patches(y, patch_size, stride) for y in ydata]:
            for patch in img:
                ypatches.append(patch)

        xpatches = np.array(xpatches)
        ypatches = np.array(ypatches)

        xpatches, ypatches = _shuffle_xy_pairs(xpatches, ypatches)

        ten_percent = int(0.1*len(xpatches))
        xtest = xpatches[:ten_percent]
        xpatches = xpatches[ten_percent:] # delete the test set

        pivot = int(split*len(xpatches))
        xtrain = xpatches[:pivot]
        xval = xpatches[pivot:]

        ytest = ypatches[:ten_percent]
        ypatches = ypatches[ten_percent:] # delete the test set

        ytrain = ypatches[:pivot]
        yval = ypatches[pivot:]

        np.save(train_filename % 'x', xtrain)
        np.save(val_filename % 'x', xval)
        np.save(test_filename % 'x', xtest)

        np.save(train_filename % 'y', ytrain)
        np.save(val_filename % 'y', yval)
        np.save(test_filename % 'y', ytest)

        return xtrain, xval, ytrain, yval
    else:
        return np.load(train_filename % 'x'), \
            np.load(val_filename % 'x'), \
            np.load(train_filename % 'y'), \
            np.load(val_filename % 'y')

def dataset(path, patch_size, stride, frac_data, batch_size, label_f):
    xtrain, xval, ytrain, yval = _maybe_create_dataset(path, patch_size, stride)

    num_examples = int(frac_data * len(xtrain))
    frac = np.random.choice(len(xtrain), num_examples)
    xtrain = xtrain[frac]
    ytrain = ytrain[frac]

    def train_iter():
        while True:
            idx = np.random.choice(len(xtrain), batch_size)
            yield xtrain[idx], np.squeeze(np.array([label_f(y) for y in ytrain[idx]]))

    return num_examples, train_iter, xval, np.squeeze(np.array([label_f(y) for y in yval]))

def tests():
    xdata, ydata = _load_data()

    x = xdata[0]
    y = ydata[0]
    assert len(_patches(x, 64, 32)) == len(_patches(y, 64, 32))

    m = np.eye(3)
    assert np.array_equal(np.array(_patches(m, 2, 1)),
                          np.array([[[1,0], [0,1]],
                                    [[0,0], [1,0]],
                                    [[0,1], [0,0]],
                                    [[1,0], [0,1]]]))

    patch_size = 10
    _maybe_create_dataset('.', patch_size=patch_size, stride=5)
    print("loading data, should not need to create new dataset")
    _maybe_create_dataset('.', patch_size=patch_size, stride=5)
    import os
    os.remove("./xtrain_patchsize" + str(patch_size) + '.npy')
    os.remove("./xval_patchsize" + str(patch_size) + '.npy'  )
    os.remove("./xtest_patchsize" + str(patch_size) + '.npy' )
    os.remove("./ytrain_patchsize" + str(patch_size) + '.npy')
    os.remove("./yval_patchsize" + str(patch_size) + '.npy'  )
    os.remove("./ytest_patchsize" + str(patch_size) + '.npy' )

if __name__ == '__main__':
    tests()

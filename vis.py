import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import cedars_sinai_etl
import resnet

# basedir = "/home/gideon/Data/cedars-sinai/"
basedir = "/mnt/data/"
img_filename = basedir + "TIFF color normalized sequential filenames/test%d.tif"
raw_label_filename = basedir + "ATmask sequential filenames/test%d_Mask.mat"
png_label_filename = basedir + "ATmask sequential filenames png/test%d_Mask.png"
with_annotations_filename = basedir + "Color annotation sequential filenames/test%d_Annotated.tif"

def plot_row_of_images(sample_num):
    '''
    Plots a row of raw_image, original annotated img, grayscale annotation mask (4 classes)
    '''
    raw_img = cv2.imread(img_filename %(sample_num))
    annotated_img = cv2.imread(with_annotations_filename %(sample_num))
    labels_img = cv2.imread(png_label_filename %(sample_num))
    assert img != None

    fig = plt.figure()

    fig.add_subplot(131)
    plt.imshow(raw_img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    fig.add_subplot(132)
    plt.imshow(annotated_img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    fig.add_subplot(133)
    plt.imshow(labels_img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()

img = cv2.imread(img_filename %(10))
assert img != None
patches = cedars_sinai_etl._patches(img_filename %(10))

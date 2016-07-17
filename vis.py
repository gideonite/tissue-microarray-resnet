import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import cedars_sinai_etl
import resnet
import sklearn.metrics
import json


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

def overlay_predictions(img, path_to_checkpoint):
    patch_size = 64
    stride = 1
    ret = []
    for x in range(int((img.shape[0]-patch_size+1)/stride)):
        patches = []
        for y in range(int((img.shape[1]-patch_size+1)/stride)):
            patch = img[x*stride:x*stride+patch_size, y*stride:y*stride+patch_size]
            patches.append(patches)

        patches = np.array(patches)
        preds = resnet.predict(patches, path_to_checkpoint)
        ret.append(preds)

    return ret

# img = cv2.imread(img_filename %(10))
# assert img != None

# overlay_predictions(img, '/mnt/data/output/foobar/foobar.checkpoint')

def confusion_matrix(path_to_checkpoint, xtest_path, ytest_path):
    xtest = np.load(xtest_path)
    ytest = np.load(ytest_path)

    preds = resnet.predict(xtest, path_to_checkpoint)
    return sklearn.metrics.confusion_matrix(ytest, preds)

# confmatrix = confusion_matrix('/mnt/data/output/foobar/foobar.checkpoint', '/mnt/data/output/xtest.npy', '/mnt/data/output/ytest.npy') 

def save_or_show(plt, filename):
    if filename != None:
        plt.savefig(filename)
    else:
        plt.show()
    
def plot_train_accs(filename, output_filename=None):
    with open(filename) as f:
        data = json.load(f)
        plt.xlim(0, sum([len(accs) for accs in data['train_accs']]))

        ys = []
        for i, train_acc in enumerate(data['train_accs']):
            for acc in train_acc:
                ys.append(acc)

        plt.plot(ys, '-', color='black')

        plt.title(data['experiment_name'])
        plt.ylabel('train accuracies')
        plt.xlabel('batch')

        save_or_show(plt, output_filename)

def plot_test_accs(filename, output_filename=None):
    with open(filename) as f:
        data = json.load(f)
        plt.ylim(0,1.1)
        plt.plot(data['test_accs'], '-')       
        plt.title(data['experiment_name'])
        plt.ylabel('test accuracy')
        plt.xlabel('epoch')
        plt.show()

        save_or_show(plt, output_filename)

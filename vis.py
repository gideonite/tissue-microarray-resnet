import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
# import cedars_sinai_etl TODO etl is dead, long live etl2
import resnet
import sklearn.metrics
import json

basedir = "/home/gideon/Data/cedars-sinai/"
# basedir = "/mnt/data/"
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

def _save_or_show(plt, filename):
    if filename != None:
        plt.savefig(filename)
    else:
        plt.show(block=False)
    
def plot_train_accs(filename, output_filename=None):
    plt.figure()
    with open(filename) as f:
        data = json.load(f)
        plt.xlim(0, sum([len(accs) for accs in data['train_accs']]))

        ys = []
        for i, train_acc in enumerate(data['train_accs']):
            for acc in train_acc:
                ys.append(1 - float(acc))

        plt.plot(ys, '-', color='black')

        plt.title(data['experiment_name'])
        plt.ylabel('train error')
        plt.xlabel('batch')

        _save_or_show(plt, output_filename)

def plot_train_val_accs(filename, output_filename=None, title=None):
    '''
    Remember that this considers accs to be recorded in one continuous
    stream with x,y pairs (iter num, train/val accuracy).
    '''
    plt.figure()
    with open(filename) as f:
        data = json.load(f)

        plt.xlim([0, len(data['train_accs'])])
        plt.ylim(0,100)

        print("final train", [a[1] for a in data['train_accs']][-1])
        print("final val", [a[1] for a in data['val_accs']][-1])

        train_plot = plt.plot([100.0 * (1.0 - float(acc[1])) for acc in data['train_accs']] , '--', color='black', label='Training Error')
        val_plot = plt.plot([100.0 * (1.0 - float(acc[1])) for acc in data['val_accs']], '-', color='red', label='Validation Error')
        plt.legend()

        if title == None:
            title = data['experiment_name']

        plt.title(title)
        plt.ylabel('error (%)')
        plt.xlabel('iteration (1e2)')

        _save_or_show(plt, output_filename)

def plot_test_accs(filename, output_filename=None):
    plt.figure()
    with open(filename) as f:
        data = json.load(f)
        plt.ylim(0,1.1)
        plt.plot([1 - float(y) for y in data['test_accs']], '-')       
        plt.title(data['experiment_name'])
        plt.ylabel('test error')
        plt.xlabel('epoch')

        _save_or_show(plt, output_filename)



if __name__ == '__main__':
    # TODO come up with a light weight cli.
    # results_titles_outputs = [['notebooks/results/2labels_75epochs.json', 'Cancer/Non-Cancer 41 Layer ResNet', 'notes/cedars_sinai_binary_41layers.png'],
    #                           ['notebooks/results/2labels_shallownet.json', 'Cancer/Non-cancer 10 Layer ResNet', 'notes/cedars_sinai_binary_10layers.png'],
    #                           ['notebooks/results/4labels_adamopt01.json', '4 Labels Using ADAM Optimizer', 'notes/cedars_sinai_4labels_adamopt.png'],
    #                           ['notebooks/results/4labels_adamopt_nodecay.json', '4 Labels Using ADAM Optimizer Without Weight Decay', 'notes/cedars_sinai_4labels_adamopt_noweightdecay.png']]

    # for filename, title, output_filename in results_titles_outputs:
    #     filename = os.path.expandvars(filename)
    #     plot_train_val_accs(filename, title=title, output_filename=output_filename)
    pass

# sample_num = 10

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np

# import scipy.io as sio
# raw_img = cv2.imread(img_filename %(sample_num))
# labels = sio.loadmat(raw_label_filename % sample_num)['ATmask']

# plt.figure(1)

# plt.subplot(131)
# imgplot = plt.imshow(raw_img)

# plt.subplot(132)
# imgplot = plt.imshow(cv2.imread(with_annotations_filename % sample_num))


# plt.subplot(133)
# # imgplot = plt.imshow(raw_img)
# plt.bar([1,2,3,4], np.bincount(labels.flatten())[1:] / sum(np.bincount(labels.flatten())), color='black')
# plt.show()

# # cv2.imshow('foobar', raw_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

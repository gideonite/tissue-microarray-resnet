import json
import tensorflow as tf
import time
import sys

import resnet
import cedars_sinai_train

flags = tf.app.flags
FLAGS = flags.FLAGS

MODEL_SAVEPATH = mkdir(FLAGS.cache_basepath + '/' + FLAGS.experiment_name)

LOG_PATH = FLAGS.results_basepath  + FLAGS.experiment_name + ".json"

def main(_):
    # create the model based on what's in the log file.
    # for each test file,
        # for each pixel in test file
            # evaluate the patch at that pixel and save in a matrix
    # pad the matrix to match dimensions of the image (dim mismatch
    # because of patch size). Basically need to patch with patchsize
    # all around the image.

    # save as .png in the MODEL_SAVEPATH folder as filename_pred.png
    
    print("HELLO WORLD")

if __name__ == '__main__':
    tf.app.run()

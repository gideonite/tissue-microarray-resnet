from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from math import sqrt
import os

# from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data

try:
    mnist
except NameError:
    mnist = input_data.read_data_sets('MNIST_data')

def resnet():
    return 42

classifier = learn.TensorFlowEstimator(
    model_fn=resnet, n_classes=10, batch_size=100, steps=100,
    learning_rate=0.001, continue_training=True, optimizer='SGD')

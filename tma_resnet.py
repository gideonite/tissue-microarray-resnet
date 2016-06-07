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

CONFIG = {'use_cudnn_on_gpu': True,
          'chip': '/gpu:0',
          'batch_size': 128,
          'num_classes': 10
}

try:
    mnist
except NameError:
    mnist = input_data.read_data_sets('MNIST_data')

def variable_on_chip(name, shape, initializer, chip=CONFIG['chip']):
    with tf.device(chip):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

def resnet(x, y, activation=tf.nn.relu): # TODO, according to paper don't use relu...
    with tf.variable_scope('layer1') as scope:
        num_filters1 = 64
        num_examples, height, width, num_channels = x.get_shape().as_list()
        convfilter = [3, 3, num_channels, num_filters1]
        weights = variable_on_chip('weights', convfilter,
                                   tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(x, weights, [1,1,1,1], padding='SAME')
        biases = variable_on_chip('biases', [num_filters1], tf.constant_initializer(0.0))

        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1') TODO local response norm?
    out1 = pool1

    with tf.variable_scope('layer2') as scope:
        num_filters2 = 64
        weights = variable_on_chip('weights', [5, 5, num_filters1, num_filters2],
                                   tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(out1, weights, [1,1,1,1], padding='SAME')
        biases = variable_on_chip('biases', [num_filters2], tf.constant_initializer(0.0))

        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)

    pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool2')
    out2 = pool2

    with tf.variable_scope('layer3') as scope:
        layer3_dim = 128
        reshape = tf.reshape(out2, [-1])
        print("out2", out2)
        print("reshape", reshape)
        input_dim = reshape.get_shape().as_list()[1]

        weights = variable_on_chip('weights', [input_dim, layer3_dim],
                                   tf.contrib.layers.xavier_initializer())

        biases = variable_on_chip('biases', [layer3_dim],
                                  tf.constant_initializer(0.0))

        dense3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('layer4') as scope:
        weights = variable_on_chip('weights', [layer3_dim, CONFIG['num_classes']],
                                   tf.contrib.layers.xavier_initializer())

        biases = variable_on_chip('biases', [CONFIG['num_classes']],
                                  tf.constant_initializer(0.0))

        pred = tf.nn.relu(tf.matmul(dense3, weights) + biases, name=scope.name)
    return learn.models.logistic_regression(pred, y)

ndim = int(sqrt(mnist.train.images.shape[1]))
x_train = mnist.train.images.reshape([-1, ndim, ndim, 1])
# net = resnet(x_train, mnist.train.labels)

classifier = learn.TensorFlowEstimator(
    model_fn=resnet, n_classes=10, batch_size=CONFIG['batch_size'], steps=100,
    learning_rate=0.001, continue_training=True, optimizer='SGD')

classifier.fit(x_train, mnist.train.labels)

tf.reset_default_graph() # TODO

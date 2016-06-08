from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from math import sqrt
import os

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

CONFIG = {'use_cudnn_on_gpu': True, # TODO
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

def inference(x, activation=tf.nn.relu): # TODO, according to paper don't use relu...
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
        squashed_dim = reduce(lambda x, y: x*y, out2.get_shape().as_list()[1:])
        reshape = tf.reshape(out2, [-1, squashed_dim])
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

        preds = tf.nn.relu(tf.matmul(dense3, weights) + biases, name=scope.name)

    return preds

def loss(preds, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, labels, name='crossentropy')
    the_loss = tf.reduce_mean(cross_entropy, name='crossentropy_mean')
    return the_loss

def train(loss, learning_rate):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def accuracy(preds, labels, k=1):
    correct = tf.nn.in_top_k(preds, labels, k)
    correct = tf.cast(correct, tf.int32)
    return tf.reduce_sum(correct)
    
with tf.Session() as sess:
    ndim = int(sqrt(mnist.train.images.shape[1]))
    x_placeholder = tf.placeholder(tf.float32, shape=(None, ndim, ndim, 1))
    y_placeholder = tf.placeholder(tf.int32, shape=(None))
    
    preds = inference(x_placeholder)
    loss_op = loss(preds, y_placeholder)
    train_op = train(loss_op, 0.01)
    accuracy_op = accuracy(preds, y_placeholder)

    # TODO saver, summary

    init = tf.initialize_all_variables()
    sess.run(init)

    x_train = mnist.train.images.reshape([-1, ndim, ndim, 1])
    y_train = mnist.train.labels
    num_examples = x_train.shape[0]
    num_batches = int(x_train.shape[0] / CONFIG['batch_size'])+1
    for batch_i in xrange(num_batches):
        start = batch_i*CONFIG['batch_size']
        end = (batch_i+1)*CONFIG['batch_size']
        x_batch = x_train[start:end]

        y_batch = mnist.train.labels[start:end]

        sess.run([train_op, loss_op], feed_dict={x_placeholder: x_batch,
                                              y_placeholder: y_batch})


        random_sample = np.random.randint(num_examples, size=(1000,))

        print("training accuracy:\t",
         sess.run(accuracy_op, feed_dict={x_placeholder: x_train[random_sample],
                                          y_placeholder: y_train[random_sample]}) / 1000.0)

        
        print("test accuracy:\t\t",
         sess.run(accuracy_op, feed_dict={x_placeholder: mnist.test.images.reshape(-1,ndim,ndim,1),
                                          y_placeholder: mnist.test.labels})  / float(mnist.test.images.shape[0]))
        print()


tf.reset_default_graph() # TODO

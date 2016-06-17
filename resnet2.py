import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt

mnist = input_data.read_data_sets('/tmp/MNIST_data')

def get_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def conv2d(x, filter_shape, num_channels, stride):
    weights = get_variable('weights', filter_shape + [x.get_shape()[-1].value, num_channels])
    conv = tf.nn.conv2d(x, weights, [1,1,1,1], padding='SAME')
    mean, variance = tf.nn.moments(conv, axes=[0,1,2])
    batch_norm = tf.nn.batch_normalization(conv, mean, variance, offset=None, scale=None, variance_epsilon=0.0001)
    return tf.nn.relu(batch_norm)

def inference(xplaceholder):
    with tf.variable_scope('layer1') as scope:
        conv1 = conv2d(xplaceholder, [3, 3], 64, [1,1,1,1])

    with tf.variable_scope('layer2') as scope:
        conv2 = conv2d(conv1, [3, 3], 64, [1,1,1,1])

    with tf.variable_scope('layer3') as scope:
        conv3 = conv2d(conv2, [3, 3], 64, [1,1,1,1])

    return conv3

with tf.Session() as sess:
    ndim = int(sqrt(mnist.train.images.shape[1]))
    xplaceholder = tf.placeholder(tf.float32, shape=(None, ndim, ndim, 1))

    print(inference(xplaceholder))

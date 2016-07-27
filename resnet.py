"""
TODO some more docs.

Reference Paper: http://arxiv.org/pdf/1512.03385.pdf
"""

from collections import namedtuple
import tensorflow as tf
from math import sqrt
import numpy as np

def _get_variable(name, shape, weight_decay=None):
    '''
    The weight decay parameter gives the scaling constant for the
    L2-loss on the parameters.
    '''
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

    if weight_decay is not None:
        wd = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', wd)

    return var

def fully_connected(x, outdim, activation=tf.nn.relu):
    indim = x.get_shape()[-1].value
    weights = _get_variable('weights', [indim, outdim], weight_decay=0.0001)
    biases = tf.get_variable('biases', [outdim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x, weights) + biases

def _conv2d(x, filter_shape, num_channels, stride):
    weights = _get_variable('weights', filter_shape + [x.get_shape()[-1].value, num_channels], weight_decay=0.0001)
    conv = tf.nn.conv2d(x, weights, stride, padding='SAME')
    mean, variance = tf.nn.moments(conv, axes=[0,1,2])
    batch_norm = tf.nn.batch_normalization(conv, mean, variance,
                                           offset=None, scale=None, variance_epsilon=0.0001)
    return tf.nn.relu(batch_norm)

def _flatten(x):
    volumn = 1
    for dim in x.get_shape()[1:]:
        volumn *= dim.value
    return tf.reshape(x, [-1, volumn])

def _get_architecture_or_fail(arch):
    try:
        return architectures[arch]
    except KeyError:
        raise KeyError, "available architecures are: " + ",".join(architectures)

def inference(xplaceholder, arch_or_groups):
    '''
    Builds the model.
    
    '''

    if type(arch_or_groups) == str:
        groups = _get_architecture_or_fail(arch_or_groups)
    elif type(arch_or_groups) == list:
        groups = arch_or_groups
    else:
        raise TypeError, arch_or_groups, type(arch_or_groups)

    # First convolution expands to 64 channels
    with tf.variable_scope('first_conv_expand_layer'):
        net = _conv2d(xplaceholder, [7, 7], 64, [1, 1, 1, 1])
    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("group_0/block_0/conv_upscale"):
        net = _conv2d(net, [1, 1], groups[0].num_filters, [1,1,1,1])

    for group_i, group in enumerate(groups):
        for block_i in range(group.num_blocks):
            name = 'group_%d/block_%d' % (group_i, block_i)

            with tf.variable_scope(name + '/conv_in'):
                conv = _conv2d(net, [1, 1], group.bottleneck_size, [1, 1, 1, 1])
                
            with tf.variable_scope(name + '/conv_bottleneck'):
                conv = _conv2d(conv, [3, 3], group.bottleneck_size, [1, 1, 1, 1])

            with tf.variable_scope(name + '/conv_out'):
                conv = _conv2d(conv, [1, 1], group.num_filters, [1, 1, 1, 1])

            net = conv + net

        try:
            next_group = groups[group_i+1]
            with tf.variable_scope('group_%d/conv_upscale' % group_i):
                net = _conv2d(net, [1, 1], next_group.num_filters, [1, 1, 1, 1])
        except IndexError:
            pass

    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1],
                   strides=[1, 1, 1, 1], padding='VALID')
    return _flatten(net)
    # net = fully_connected(net, num_classes)

BottleneckGroup = namedtuple(
    'BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size'])

architectures = { '10_layers': [BottleneckGroup(3,128,32)],
                   '41_layers': [BottleneckGroup(3, 128, 32),
                                 BottleneckGroup(3, 256, 64),
                                 BottleneckGroup(3, 512, 128),
                                 BottleneckGroup(3, 1024, 256)],
                   '50_layers': [BottleneckGroup(3, 64, 256),
                                 BottleneckGroup(4, 128, 512),
                                 BottleneckGroup(6, 256, 1024),
                                 BottleneckGroup(3, 512, 2048)]}

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train_ops(xplaceholder,
              yplaceholder,
              architecture,
              num_classes):
    '''
    Returns the TF ops that you can use during training.
    '''

    arch = _get_architecture_or_fail(architecture)

    net = inference(xplaceholder, arch)
    preds = fully_connected(net, num_classes)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, yplaceholder, name='crossentropy')
    avg_cross_entropy = tf.reduce_mean(cross_entropy)
    regularization = tf.get_collection('regularized_losses')
    if regularization:
        regularization = tf.add_n(regularization)
        loss = tf.add(avg_cross_entropy, regularization)
    else:
        loss = avg_cross_entropy

    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=0.9)
    train_op = train_op.minimize(loss, global_step=global_step)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds,1),
                                               yplaceholder), tf.float32))

    return global_step, train_op, learning_rate, preds, loss, accuracy

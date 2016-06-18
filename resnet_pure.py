from collections import namedtuple
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt

mnist = input_data.read_data_sets('/tmp/MNIST_data')

def get_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def fully_connected(x, outdim, activation=tf.nn.relu):
    indim = x.get_shape()[-1].value
    weights = get_variable('weights', [indim, outdim])
    biases = tf.get_variable('biases', [outdim], initializer=tf.constant_initializer(0.0))
    return activation(tf.matmul(x, weights) + biases)

def conv2d(x, filter_shape, num_channels, stride):
    weights = get_variable('weights', filter_shape + [x.get_shape()[-1].value, num_channels])
    conv = tf.nn.conv2d(x, weights, stride, padding='SAME')
    mean, variance = tf.nn.moments(conv, axes=[0,1,2])
    batch_norm = tf.nn.batch_normalization(conv, mean, variance,
                                           offset=None, scale=None, variance_epsilon=0.0001)
    return tf.nn.relu(batch_norm)

def flatten(x):
    volumn = 1
    for dim in x.get_shape()[1:]:
        volumn *= dim.value
    return tf.reshape(x, [-1, volumn])

# Configurations for each bottleneck group.
BottleneckGroup = namedtuple(
    'BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size'])
groups = [BottleneckGroup(3, 128, 32),
          BottleneckGroup(3, 256, 64),
          BottleneckGroup(3, 512, 128),
          BottleneckGroup(3, 1024, 256)
]

def inference(xplaceholder, num_classes):
  # First convolution expands to 64 channels
    with tf.variable_scope('first_conv_expand_layer'):
        net = conv2d(xplaceholder, [7, 7], 64, [1, 1, 1, 1])
    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    for group_i, group in enumerate(groups):
        for block_i in range(group.num_blocks):
            name = 'group_%d/block_%d' % (group_i, block_i)
            
            with tf.variable_scope(name + '/conv_in'):
                conv = conv2d(net, [1, 1], group.bottleneck_size, [1, 1, 1, 1])
                
            with tf.variable_scope(name + '/conv_bottleneck'):
                conv = conv2d(net, [3, 3], group.bottleneck_size, [1, 1, 1, 1])

            with tf.variable_scope(name + '/conv_out'):
                outdim = net.get_shape()[-1].value
                conv = conv2d(net, [3, 3], outdim, [1, 1, 1, 1])

            net = conv + net
        try:
            next_group = groups[group_i+1]
            with tf.variable_scope('block_%d/conv_upscale' % group_i):
                net = conv2d(net, [1, 1], next_group.num_filters, [1, 1, 1, 1])
        except IndexError:
            pass

    net_shape = net.get_shape().as_list()
    net = tf.nn.avg_pool(net, ksize=[1, net_shape[1], net_shape[2], 1],
                   strides=[1, 1, 1, 1], padding='VALID')
    net = flatten(net)
    net = fully_connected(net, num_classes)
    return net

def train_ops(xplaceholder,
          yplaceholder,
          num_classes,
          optimizer=tf.train.GradientDescentOptimizer,
          learning_rate=0.01):

    preds = inference(xplaceholder, num_classes)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, yplaceholder,
                                                                   name='crossentropy')
    avg_loss = tf.reduce_mean(cross_entropy, name='batchwise_avg_loss')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer(learning_rate).minimize(avg_loss, global_step=global_step)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds,1),
                                               yplaceholder), tf.float32))

    return train_op, preds, avg_loss, accuracy

    # init = tf.initialize_all_variables()
    # sess.run(init)
    # num_examples = xtrain.shape[0]
    # for epoch_i in xrange(num_epochs):
    #     train_accs = []
    #     for batch_i in xrange(0, num_examples, batch_size):
    #         xbatch = xtrain[batch_i : batch_i + batch_size]
    #         ybatch = ytrain[batch_i : batch_i + batch_size]

    #         _,_,train_acc = sess.run([train_op, avg_loss, accuracy],
    #                            feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})

    #         train_accs.append(train_acc)

        # test_acc = 0
        # for batch_i in xrange(0, xtest.shape[0], batch_size):
        #     xbatch = xtest[batch_i : batch_i + batch_size]
        #     ybatch = ytest[batch_i : batch_i + batch_size]

        #     _,_,train_acc = sess.run([train_op, avg_loss, accuracy],
        #                        feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})

        #     test_acc += sess.run(accuracy, feed_dict={xplaceholder: xtest, yplaceholder: ytest})

        # yield train_accs
        # , test_acc / xtrain.shape[0]

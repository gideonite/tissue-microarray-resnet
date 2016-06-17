from collections import namedtuple
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
    batch_norm = tf.nn.batch_normalization(conv, mean, variance,
                                           offset=None, scale=None, variance_epsilon=0.0001)
    return tf.nn.relu(batch_norm)

# Configurations for each bottleneck group.
BottleneckGroup = namedtuple(
    'BottleneckGroup', ['num_blocks', 'num_filters', 'bottleneck_size'])
groups = [BottleneckGroup(3, 128, 32),
        BottleneckGroup(3, 256, 64),
        BottleneckGroup(3, 512, 128),
        BottleneckGroup(3, 1024, 256)
]

def inference(xplaceholder):
    with tf.variable_scope('layer1') as scope:
        conv1 = conv2d(xplaceholder, [3, 3], 64, [1,1,1,1])

    with tf.variable_scope('layer2') as scope:
        conv2 = conv2d(conv1, [3, 3], 64, [1,1,1,1])

    with tf.variable_scope('layer3') as scope:
        conv3 = conv2d(conv2, [3, 3], 64, [1,1,1,1])

    with tf.variable_scope('skip_connection') as scope:
        skip_connection = conv2d(xplaceholder, [1, 1], 64, [1,1,1,1]) + conv3

    with tf.variable_scope('fully_connected') as scope:
        volumn = 1
        for dim in skip_connection.get_shape()[1:]:
            volumn *= dim.value
        reshape = tf.reshape(skip_connection, [-1, volumn])

        weights = get_variable('weights', [reshape.get_shape()[-1].value, 10])
        biases = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.0))

        return tf.nn.relu(tf.matmul(reshape, weights) + biases)

with tf.Session() as sess:
    batch_size = 128
    ndim = int(sqrt(mnist.train.images.shape[1]))
    xplaceholder = tf.placeholder(tf.float32, shape=(batch_size, ndim, ndim, 1))
    yplaceholder = tf.placeholder(tf.int32, shape=(batch_size))

    preds = inference(xplaceholder)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, yplaceholder, name='crossentropy')
    avg_loss = tf.reduce_mean(cross_entropy, name='batchwise_avg_loss')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(avg_loss, global_step=global_step)

    init = tf.initialize_all_variables()
    sess.run(init)

    xtrain = mnist.train.images.reshape([-1, 28, 28, 1])
    ytrain = mnist.train.labels
    num_examples = xtrain.shape[0]

    for batch_i in xrange(0, num_examples, batch_size):
        xbatch = xtrain[batch_i : batch_i + batch_size]
        ybatch = ytrain[batch_i : batch_i + batch_size]

        t, a = sess.run([train_op, avg_loss], feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})

        # print("training loss: ", a)

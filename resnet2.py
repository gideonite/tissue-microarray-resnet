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

def inference(xplaceholder):
    with tf.variable_scope('layer1') as scope:
        net = conv2d(xplaceholder, [3, 3], 64, [1,1,1,1])

    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer2') as scope:
        net = conv2d(net, [3, 3], 64, [1,1,1,1])

    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3') as scope:
        net = conv2d(net, [3, 3], 64, [1,1,1,1])

    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('skip_connection') as scope:
        net = conv2d(xplaceholder, [7, 7], 64, [1,7,7,1]) + net

    with tf.variable_scope('fully_connected1') as scope:
        flattened = flatten(net)
        net = fully_connected(flattened, 1000)

    return net

with tf.Session() as sess:
    batch_size = 128
    ndim = int(sqrt(mnist.train.images.shape[1]))
    xplaceholder = tf.placeholder(tf.float32, shape=(None, ndim, ndim, 1))
    yplaceholder = tf.placeholder(tf.int64, shape=(None))

    preds = inference(xplaceholder)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, yplaceholder, name='crossentropy')
    avg_loss = tf.reduce_mean(cross_entropy, name='batchwise_avg_loss')

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds,1), yplaceholder), tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(avg_loss, global_step=global_step)

    init = tf.initialize_all_variables()
    sess.run(init)

    xtrain = mnist.train.images.reshape([-1, 28, 28, 1])
    ytrain = mnist.train.labels
    num_examples = xtrain.shape[0]

    for epoch in xrange(50):
        losses = []
        accs = []
        for batch_i in xrange(0, num_examples, batch_size):
            xbatch = xtrain[batch_i : batch_i + batch_size]
            ybatch = ytrain[batch_i : batch_i + batch_size]

            _, batchloss, a = sess.run([train_op, avg_loss, accuracy], feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})
            losses.append(batchloss)
            accs.append(a)

            # print("training accuracy: " + str(a))

        print("epoch: " + str(epoch) + " " + " avg loss: " + str(sum(losses) / len(losses)) + " avg tr accuracy: " + str(sum(accs) / len(accs)))

        # print("training loss: ", a)

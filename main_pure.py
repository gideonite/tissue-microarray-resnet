from sklearn import metrics
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import json
from math import sqrt
import time

import cedars_sinai_etl
import resnet_pure

flags = tf.app.flags
FLAGS = flags.FLAGS

timestamp = str(time.time())

flags.DEFINE_string('cache_basepath', '/mnt/data/', '')
flags.DEFINE_string('results_basepath', '/mnt/code/notebooks/results', '')
flags.DEFINE_string('experiment_name', str(timestamp), '')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_integer('batch_size', 64, 'Number of examples per GD batch')

def main(_):
    xdata, ydata = cedars_sinai_etl.dataset(num_images=5)
    ndim = int(sqrt(xdata.shape[1] / 3))
    xdata = xdata.reshape(-1, ndim, ndim, 3)

    xtrain, xtest, ytrain, ytest = train_test_split(
        xdata, ydata, test_size=0.2, random_state=42)

    log = {'train_accs': [],
           'test_accs': [],
           'num_epochs': 20,
           'timestamp': timestamp,
           'batch_size': FLAGS.batch_size,
           'patch_size': FLAGS.patch_size,
           'stride': FLAGS.stride}

    example = xtrain[0]
    assert example.shape[0] == example.shape[1]
    ndim = example.shape[0]
    num_channels = example.shape[2]
    xplaceholder = tf.placeholder(tf.float32, shape=(None, ndim, ndim, num_channels))
    yplaceholder = tf.placeholder(tf.int64, shape=(None))
    trainer, preds, loss, accuracy = resnet_pure.train_ops(xplaceholder, yplaceholder, num_classes=4)
    num_examples = xtrain.shape[0]
    with tf.Session() as sess:
        for epoch_i in xrange(FLAGS.num_epochs):
            train_accs = []
            for batch_i in xrange(0, num_examples, FLAGS.batch_size):
                xbatch = xtrain[batch_i : batch_i + FLAGS.batch_size]
                ybatch = ytrain[batch_i : batch_i + FLAGS.batch_size]
                
                # train_loss, train_acc = sess.run([trainer, loss, accuracy],
                #                          feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})

    # with tf.Session() as sess:
    #     for train_accs in resnet_pure.train(xtrain,
    #                                         ytrain,
    #                                         xtest,
    #                                         ytest,
    #                                         4,
    #                                         FLAGS.batch_size,
    #                                         FLAGS.num_epochs,
    #                                         sess,
    #                                         optimizer=tf.train.GradientDescentOptimizer,
    #                                         learning_rate=0.01):
    #         log['train_accs'].append(train_accs)

    #         print(test_acc)

    #         # overwrite the log file each time.
    #         with open(FLAGS.training_log_dir, 'w+') as logfile:
    #             json.dump(log, logfile)
        
if __name__ == '__main__':
    tf.app.run()

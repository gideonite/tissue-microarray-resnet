from sklearn.cross_validation import train_test_split
import tensorflow as tf
import json
from math import sqrt
import time
import sys
import os.path

import cedars_sinai_etl
import resnet_pure

flags = tf.app.flags
FLAGS = flags.FLAGS

timestamp = str(time.time())

flags.DEFINE_string('cache_basepath', '/mnt/data/models/', '')
flags.DEFINE_string('results_basepath', '/mnt/code/notebooks/results/', '')
flags.DEFINE_string('experiment_name', 'experiment_' + str(timestamp), '')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_integer('batch_size', 64, 'Number of examples per GD batch')
flags.DEFINE_integer('num_images', -1, '')
flags.DEFINE_boolean('clobber', False, 'Start training from scratch or not')

def maybe_load_logfile(path):
    if os.path.exists(path) and not FLAGS.clobber:
        with open(path) as f:
            log = json.load(f)
    else:
        print("creating new experiment from scratch in '" + path + "'")
        log = {'cmd': " ".join(sys.argv), # TODO, want to separate cmd line args from code to automatically restart experiments.
               'architecture': resnet_pure.groups,
               'train_accs': [],
               'test_accs': [],
               'num_epochs': FLAGS.num_epochs,
               'num_images': FLAGS.num_images,
               'timestamp': timestamp,
               'experiment_name': FLAGS.experiment_name,
               'batch_size': FLAGS.batch_size,
               'patch_size': FLAGS.patch_size,
               'stride': FLAGS.stride}

    return log

def main(_):
    if FLAGS.num_images != -1:
        xdata, ydata = cedars_sinai_etl.dataset(num_images=FLAGS.num_images)
    else:
        xdata, ydata = cedars_sinai_etl.dataset()

    ndim = int(sqrt(xdata.shape[1] / 3))
    xdata = xdata.reshape(-1, ndim, ndim, 3)

    xtrain, xtest, ytrain, ytest = train_test_split(
        xdata, ydata, test_size=0.2, random_state=42)

    resultspath = FLAGS.results_basepath  + FLAGS.experiment_name + ".json"
    log = maybe_load_logfile(resultspath)
    json.dump(log, sys.stdout, indent=2)

    example = xtrain[0]
    assert example.shape[0] == example.shape[1]
    ndim = example.shape[0]
    num_channels = example.shape[2]
    xplaceholder = tf.placeholder(tf.float32, shape=(None, ndim, ndim, num_channels))
    yplaceholder = tf.placeholder(tf.int64, shape=(None))
    train_step, preds, loss, accuracy = resnet_pure.train_ops(xplaceholder, yplaceholder, optimizer=tf.train.AdamOptimizer, num_classes=4)
    num_examples = xtrain.shape[0]
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    savepath = FLAGS.cache_basepath + FLAGS.experiment_name + ".checkpoint"
    with tf.Session() as sess:
        sess.run(init)
        try:
            if not FLAGS.clobber:
                saver.restore(sess, savepath)
            else:
                raise ValueError # TODO hack
        except ValueError:
            # test on the randomly intialized model
            test_accs = []
            for batch_i in xrange(0, xtest.shape[0], FLAGS.batch_size):
                xbatch = xtest[batch_i : batch_i + FLAGS.batch_size]
                ybatch = ytest[batch_i : batch_i + FLAGS.batch_size]
                test_accs.append(sess.run(accuracy, feed_dict={xplaceholder: xbatch, yplaceholder: ybatch}))
            test_acc = sum(test_accs) / len(test_accs)
            log['test_accs'].append(str(test_acc))
            print("\n%s\t epoch: 0 test_accuracy=%f" %(FLAGS.experiment_name, test_acc))

        for epoch_i in xrange(FLAGS.num_epochs):
            train_accs = []
            for batch_i in xrange(0, num_examples, FLAGS.batch_size):
                xbatch = xtrain[batch_i : batch_i + FLAGS.batch_size]
                ybatch = ytrain[batch_i : batch_i + FLAGS.batch_size]

                # TODO optimize, don't need to track acc if you are already tracking loss.
                _, train_loss, train_acc = sess.run([train_step, loss, accuracy],
                                                 feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})

                # stringify for JSON serialization
                train_accs.append(str(train_acc))

                print("epoch: %d batch: %d training_accuracy=%f" %(epoch_i+1, batch_i/FLAGS.batch_size, train_acc))

            saver.save(sess, savepath)

            test_accs = []
            for batch_i in xrange(0, xtest.shape[0], FLAGS.batch_size):
                xbatch = xtest[batch_i : batch_i + FLAGS.batch_size]
                ybatch = ytest[batch_i : batch_i + FLAGS.batch_size]
                test_accs.append(sess.run(accuracy, feed_dict={xplaceholder: xbatch, yplaceholder: ybatch}))

            log['train_accs'].append(train_accs)
            test_acc = sum(test_accs) / len(test_accs)
            log['test_accs'].append(str(test_acc))
            print("%s\t epoch: %d test_accuracy=%f" %(FLAGS.experiment_name, epoch_i+1, test_acc))

            with open(resultspath, 'w+') as logfile:
                json.dump(log, logfile, indent=2)

if __name__ == '__main__':
    tf.app.run()

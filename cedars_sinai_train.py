import json
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from math import sqrt
import os.path
import pickle
import time
import sys

import cedars_sinai_etl
import resnet
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

timestamp = str(time.time())

flags.DEFINE_string('cache_basepath', '/mnt/data/output/', '')
flags.DEFINE_string('results_basepath', '/mnt/code/notebooks/results/', '')
flags.DEFINE_string('experiment_name', 'experiment_' + str(timestamp), '')
flags.DEFINE_float('frac_data', 1.0, 'Fraction of training data to use')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_integer('batch_size', 64, 'Number of examples per GD batch')
flags.DEFINE_boolean('clobber', False, 'Start training from scratch or not')
flags.DEFINE_boolean('debug', False, 'Run in debug mode. (Skips test set evaluation).')

def maybe_load_logfile(path):
    if os.path.exists(path) and not FLAGS.clobber:
        with open(path) as f:
            log = json.load(f)
    else:
        print("creating new experiment from scratch in '" + path + "'")
        log = {'cmd': " ".join(sys.argv), # TODO, want to separate cmd line args from code to automatically restart experiments.
               'architecture': [g._asdict() for g in resnet.groups],
               'train_accs': [],
               'test_accs': [],
               'num_epochs': FLAGS.num_epochs,
               'num_epochs_completed': 0,
               'timestamp': timestamp,
               'experiment_name': FLAGS.experiment_name,
               'batch_size': FLAGS.batch_size,
               'patch_size': FLAGS.patch_size,
               'stride': FLAGS.stride}

    return log

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
         
    return path

printable_params = set(['architecture', 'num_epochs', 'num_epochs_completed',\
                        'timestamp', 'experiment_name', 'batch_size',\
                        'patch_size', 'stride'])

def main(_):
    xtrain, xtest, ytrain, ytest = cedars_sinai_etl.dataset(path=FLAGS.cache_basepath)
    xtrain = xtrain[:int(len(xtrain)*FLAGS.frac_data),:]
    ytrain = ytrain[:int(len(ytrain)*FLAGS.frac_data)]

    ndim = int(sqrt(xtrain.shape[1] / 3))
    xtrain = xtrain.reshape(-1, ndim, ndim, 3)
    xtest = xtest.reshape(-1, ndim, ndim, 3)

    resultspath = FLAGS.results_basepath  + FLAGS.experiment_name + ".json"
    log = maybe_load_logfile(resultspath)
    log['num_training_examples'] = len(xtrain)
    log['num_test_examples'] = len(xtest)

    json.dump(dict((k,v) for k,v in log.iteritems() if k in printable_params),\
              sys.stdout, indent=2)

    example = xtrain[0]
    assert example.shape[0] == example.shape[1]
    ndim = example.shape[0]
    num_channels = example.shape[2]
    xplaceholder = tf.placeholder(tf.float32, shape=(None, ndim, ndim, num_channels))
    yplaceholder = tf.placeholder(tf.int64, shape=(None))
    train_step, predictor, loss, accuracy = resnet.train_ops(xplaceholder, yplaceholder, optimizer=tf.train.AdamOptimizer, num_classes=4)
    num_examples = xtrain.shape[0]
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    savepath = mkdir(FLAGS.cache_basepath + '/' + FLAGS.experiment_name) \
        + '/' + FLAGS.experiment_name + '.checkpoint'

    with tf.Session() as sess:
        sess.run(init)
        try:
            if not FLAGS.clobber:
                saver.restore(sess, savepath)
            else:
                raise ValueError # TODO hack
        except ValueError:
            if not FLAGS.debug:
                # test on the randomly intialized model, but skip this in debug mode and get straight to training.
                test_accs = []
                for batch_i in xrange(0, len(xtest), FLAGS.batch_size):
                    xbatch = xtest[batch_i : batch_i + FLAGS.batch_size]
                    ybatch = ytest[batch_i : batch_i + FLAGS.batch_size]
                    test_accs.append(sess.run(accuracy, feed_dict={xplaceholder: xbatch, yplaceholder: ybatch}))
                test_acc = sum(test_accs) / len(test_accs)
                log['test_accs'].append(str(test_acc))
                print("\n%s\t epoch: 0 test_accuracy=%f" %(FLAGS.experiment_name, test_acc))

        for epoch_i in xrange(FLAGS.num_epochs):
            # shuffle training data for each epoch
            idx = np.array(list(range(len(xtrain))))
            np.random.shuffle(idx)
            xtrain = np.array(xtrain)[idx]
            ytrain = np.array(ytrain)[idx]

            # training
            train_accs = []
            for batch_i in xrange(0, num_examples, FLAGS.batch_size):
                xbatch = xtrain[batch_i : batch_i + FLAGS.batch_size]
                ybatch = ytrain[batch_i : batch_i + FLAGS.batch_size]

                # TODO optimize, don't need to track acc if you are already tracking loss.
                _, train_loss, train_acc = sess.run([train_step, loss, accuracy],
                                                 feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})

                # stringify for JSON serialization
                train_accs.append(str(train_acc))

                print("epoch: %d/%d batch: %d training_accuracy=%f" \
                      %(epoch_i+1, FLAGS.num_epochs, batch_i/FLAGS.batch_size, train_acc))

            # save the model immediately
            saver.save(sess, savepath)

            # testing
            test_accs = []
            predictions = []
            for batch_i in xrange(0, len(xtest), FLAGS.batch_size):
                xbatch = xtest[batch_i : batch_i + FLAGS.batch_size]
                ybatch = ytest[batch_i : batch_i + FLAGS.batch_size]

                acc, preds = sess.run([accuracy, predictor], \
                                      feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})
                
                test_accs.append(acc)
                predictions.append(preds)

            log['test_predictions'] = pickle.dumps(np.concatenate(predictions))
            log['train_accs'].append(train_accs)
            test_acc = sum(test_accs) / len(test_accs)
            log['test_accs'].append(str(test_acc))
            log['num_epochs_completed'] += 1
            print("%s\t epoch: %d test_accuracy=%f" %(FLAGS.experiment_name, epoch_i+1, test_acc))

            with open(resultspath, 'w+') as logfile:
                try:
                    json.dump(log, logfile, indent=4)
                except TypeError, e:
                    print(log)
                    raise TypeError, e

if __name__ == '__main__':
    tf.app.run()

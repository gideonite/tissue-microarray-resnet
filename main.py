import argparse
from cedars_sinai_etl import dataset
import numpy as np
from resnet import res_net
import tensorflow as tf

from sklearn import metrics
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from tensorflow.contrib import learn
import json
import time

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_cache_dir', '/mnt/data/models/', 'Directory to save models and summaries')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_string('optimizer', 'SGD', 'Which optimizer to use for GD. Choose from SGD, Adagrad, ADAM')
flags.DEFINE_integer('batch_size', 64, 'Number of examples per GD batch')

def main(_):
    classifier = learn.TensorFlowEstimator(
        model_fn=res_net, n_classes=4, batch_size=FLAGS.batch_size, steps=100,
        learning_rate=0.001, continue_training=True)

    xdata, ydata = dataset(num_images=50)

    xtrain, xtest, ytrain, ytest = train_test_split(
        xdata, ydata, test_size=0.2, random_state=42)

    num_examples = xtrain.shape[0]

    log = {'num_epochs': 20,
           'timestamp': time.time(),
           'batch_size': FLAGS.batch_size,
           'patch_size': FLAGS.patch_size,
           'stride': FLAGS.stride}

    print(log)
    
    accuracies = []
    log['accuracies'] = accuracies
    for epoch in xrange(FLAGS.num_epochs):
        accuracies.append([])
        for batch_i in xrange(num_examples / FLAGS.batch_size):
            xbatch = xtrain[batch_i * FLAGS.batch_size : (batch_i+1) * FLAGS.batch_size,:]
            ybatch = ytrain[batch_i * FLAGS.batch_size : (batch_i+1) * FLAGS.batch_size]
            classifier.fit(xbatch, ybatch, logdir=FLAGS.model_cache_dir)

            tr_accuracy = metrics.accuracy_score(
                ytrain, classifier.predict(xtrain, batch_size=100))
            te_accuracy = metrics.accuracy_score(
                ytest, classifier.predict(xtest, batch_size=100))

            accuracies[-1].append({'train_accuracy': tr_accuracy, 'test_accuracy': te_accuracy})

        # last batch
        xbatch = xtrain[(batch_i+1) * FLAGS.batch_size:, :]
        ybatch = ytrain[(batch_i+1) * FLAGS.batch_size:]
        classifier.fit(xbatch, ybatch, logdir=FLAGS.model_cache_dir)
        tr_accuracy = metrics.accuracy_score(
            ytrain, classifier.predict(xtrain, batch_size=100))
        te_accuracy = metrics.accuracy_score(
            ytest, classifier.predict(xtest, batch_size=100))
        accuracies[-1].append({'train_accuracy': tr_accuracy, 'test_accuracy': te_accuracy})

        with open(FLAGS.model_cache_dir + '/training-history.json') as logfile:
            json.dump(log, logfile)

        classifier.save(FLAGS.model_cache_dir)

        print('Epoch: ' + str(epoch))
        print('Accuracy: {0:f}'.format(score))
        print('Training Error: {0:f}'.format(tr_error))
        
def reset_flags():
    '''
    Courtesy of https://gist.github.com/wookayin/6ca93b8309f05055c6a3
    '''
    tf.app.flags.FLAGS = tf.python.platform.flags._FlagValues()
    tf.app.flags._global_parser = argparse.ArgumentParser()

if __name__ == '__main__':
    tf.app.run()

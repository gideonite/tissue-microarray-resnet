import argparse
from cedars_sinai_etl import dataset
import numpy as np
from resnet import res_net
import tensorflow as tf

from sklearn import metrics
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from tensorflow.contrib import learn

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_cache_dir', '/mnt/data/models/cedars-sinai-resnet', 'Directory to save models and summaries')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_string('optimizer', 'SGD', 'Which optimizer to use for GD. Choose from SGD, Adagrad, ADAM')

def main(_):
    classifier = learn.TensorFlowEstimator(
        model_fn=res_net, n_classes=4, batch_size=100, steps=100,
        learning_rate=0.001, continue_training=True)

    xdata, ydata = dataset(num_images=50, train_test_split=1)[:2] # TODO

    xtrain, xtest, ytrain, ytest = train_test_split(
        xdata, ydata, test_size=0.2, random_state=42)
        
    for epoch in xrange(FLAGS.num_epochs):
        classifier.fit(
            xtrain, ytrain, logdir=FLAGS.model_cache_dir)

        # Calculate accuracy and training error
        score = metrics.accuracy_score(
            ytest, classifier.predict(xtest, batch_size=100))
        tr_error = metrics.accuracy_score(
            ytrain, classifier.predict(xtrain, batch_size=100))
        print('Accuracy: {0:f}'.format(score))
        print('Training Error: {0:f}'.format(tr_error))
        with open(FLAGS.model_cache_dir + 'training.log', 'a') as accuracy_file:
            accuracy_file.write('Accuracy: {0:f}'.format(score))
            accuracy_file.write('\n')

            accuracy_file.write('Training Error: {0:f}'.format(tr_error))
            accuracy_file.write('\n')

        # classifier.save(FLAGS.model_cache_dir)

def reset_flags():
    '''
    Courtesy of https://gist.github.com/wookayin/6ca93b8309f05055c6a3
    '''
    tf.app.flags.FLAGS = tf.python.platform.flags._FlagValues()
    tf.app.flags._global_parser = argparse.ArgumentParser()

if __name__ == '__main__':
    tf.app.run()

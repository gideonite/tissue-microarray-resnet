from sklearn import metrics
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import json
from math import sqrt

import cedars_sinai_etl
import resnet_pure

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_cache_dir', '/mnt/data/models/', 'Directory to save models and summaries')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_integer('batch_size', 64, 'Number of examples per GD batch')

def main(_):
    xdata, ydata = cedars_sinai_etl.dataset(num_images=5)
    ndim = int(sqrt(xdata.shape[1] / 3))
    xdata = xdata.reshape(-1, ndim, ndim, 3)

    print(xdata.shape)

    with tf.Session() as sess:
        for accs in resnet_pure.train(xdata, ydata,
                                      4,
                                      FLAGS.batch_size,
                                      FLAGS.num_epochs,
                                      sess,
                                      optimizer=tf.train.GradientDescentOptimizer,
                                      learning_rate=0.01):
            print(accs)
        
if __name__ == '__main__':
    tf.app.run()

import json
import tensorflow as tf
from math import sqrt
import os.path
import pickle
import time
import sys

import cedars_sinai_etl as etl
import resnet
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

timestamp = str(time.time())

flags.DEFINE_string('cache_basepath', '/mnt/data/output/', '')
flags.DEFINE_string('results_basepath', '/mnt/code/notebooks/results/', '')
flags.DEFINE_string('experiment_name', 'experiment_' + str(timestamp), '')
flags.DEFINE_string('architecture', '41_layers', '')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_integer('batch_size', 64, 'Number of examples per GD batch')
flags.DEFINE_boolean('clobber', False, 'Start training from scratch or not')
flags.DEFINE_boolean('debug', False, 'Run in debug mode. (Skips test set evaluation).')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs to use for training and testing.')
flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
TOWER_NAME = 'tower'

def maybe_load_logfile(path):
    if os.path.exists(path) and not FLAGS.clobber:
        with open(path) as f:
            log = json.load(f)
    else:
        print("creating new experiment from scratch in '" + path + "'")
        log = {'cmd': " ".join(sys.argv), # TODO, want to separate cmd line args from code to automatically restart experiments.
               'architecture': [g._asdict() for g in resnet.architectures[FLAGS.architecture]],
               'train_accs': [],
               'val_accs': [],
               'num_epochs': FLAGS.num_epochs,
               'num_epochs_completed': 0,
               'timestamp': timestamp,
               'experiment_name': FLAGS.experiment_name,
               'batch_size': FLAGS.batch_size,
               'patch_size': FLAGS.patch_size,
               'stride': FLAGS.stride}

    return log

printable_params = set(['architecture', 'num_epochs', 'num_epochs_completed',\
                        'timestamp', 'experiment_name', 'batch_size',\
                        'patch_size', 'stride'])

def run_validation(sess, accuracy, xplaceholder, yplaceholder, xval, yval):
    accs = []
    for batch_i in xrange(0, len(xval), FLAGS.batch_size):
        xbatch = xval[batch_i : batch_i + FLAGS.batch_size]
        ybatch = yval[batch_i : batch_i + FLAGS.batch_size]
        accs.append(sess.run(accuracy, feed_dict={xplaceholder: xbatch, yplaceholder: ybatch}))
    acc = sum(accs) / len(accs)
    return acc

def savelog(resultspath, log):
    with open(resultspath, 'w+') as logfile:
        try:
            json.dump(log, logfile, indent=4)
        except TypeError, e:
            print(log)
            raise TypeError, e

def next_learning_rate(lr, curr_epoch):
    if 0 <= curr_epoch <= 5:
        return lr
    elif 5 < curr_epoch <= 10:
        return lr / 10.0
    else:
        return lr / 100.0

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def maybe_restore_model(sess, saver):
    savepath = mkdir(FLAGS.cache_basepath + '/' + FLAGS.experiment_name) \
        + '/' + FLAGS.experiment_name + '.checkpoint'

    try:
        if not FLAGS.clobber:
            saver.restore(sess, savepath)
            return True
    except ValueError:
        pass

    return False

def tower_loss(scope, xplaceholder, yplaceholder):
    
    net = resnet.inference(xplaceholder, FLAGS.architecture)
    num_labels = 4
    logits = resnet.fully_connected(net, outdim=num_labels)
    _ = resnet.loss(logits, yplaceholder)
    
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')

    # TODO required for syncing, I think.
    # with tf.control_dependencies([loss_averages_op]):
    #     total_loss = tf.identity(total_loss)
    return total_loss

def _average_grads(tower_grads):
    '''
    Taken from the models/images/cifar10 example in the TF source.
    '''
    average_grads = []
    for grad_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_vars:
            g = tf.expand_dims(g, 0)
            grads.append(g)

        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        _, var = grad_vars[0]
        average_grads.append((grad, var))

    return average_grads

def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        learning_rate = tf.placeholder(tf.float32, shape=[])

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9)

        num_channels = 3
        tower_grads = []
        # xplaceholder = tf.placeholder(tf.float32, shape=(None, FLAGS.patch_size, FLAGS.patch_size, num_channels))
        # yplaceholder = tf.placeholder(tf.int64, shape=(None))

        # placeholders = [
        #     [tf.placeholder(tf.float32, shape=(None, FLAGS.patch_size, FLAGS.patch_size, num_channels)),
        #      tf.placeholder(tf.int64, shape=(None))],
        #     [tf.placeholder(tf.float32, shape=(None, FLAGS.patch_size, FLAGS.patch_size, num_channels)),
        #      tf.placeholder(tf.int64, shape=(None))]]

        placeholders = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' %(TOWER_NAME, i)) as scope:
                    xplaceholder = tf.placeholder(tf.float32, shape=(None, FLAGS.patch_size, FLAGS.patch_size, num_channels), name='xplaceholder_gpu%d' % i)
                    yplaceholder = tf.placeholder(tf.int64, shape=(None), name='yplaceholder_gpu%d' % i)
                    placeholders.append([xplaceholder, yplaceholder])
                    
                    loss = tower_loss(scope, xplaceholder, yplaceholder)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    tf.get_variable_scope().reuse_variables()

        grads = _average_grads(tower_grads)

        train_op = optimizer.apply_gradients(grads)

        sess = tf.Session()
        
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        init = tf.initialize_all_variables()
        sess.run(init)

        num_examples, train_iter, xval, yval = etl.dataset(path=FLAGS.cache_basepath,
                                                           patch_size=FLAGS.patch_size,
                                                           stride=FLAGS.stride,
                                                           batch_size=FLAGS.batch_size,
                                                           frac_data=FLAGS.frac_data,
                                                           label_f=etl.center_pixel)

        for i in range(10):
            it = train_iter()
            xbatch1, ybatch1 = next(it)
            xbatch2, ybatch2 = next(it)

            feed_dict = {placeholders[0][0]: xbatch1, placeholders[0][1]: ybatch1,
                        placeholders[1][0]: xbatch2, placeholders[1][1]: ybatch2, learning_rate: 0.1 }

            # feed_dict = {placeholders[0][0]: xbatch1, placeholders[0][1]: ybatch1}

            ret = sess.run([train_op], feed_dict=feed_dict)

            print(i, ret)



        # for xbatch, ybatch in train_iter():
        #     sess.run([train_op], feed_dict={xplaceholder:xbatch, yplaceholder:ybatch, learning_rate: 0.1})
        #     # accounting()

        sess.close()
            
def main(_):
    train()
    # num_examples, train_iter, xval, yval = etl.dataset(path=FLAGS.cache_basepath,
    #                                                    patch_size=FLAGS.patch_size,
    #                                                    stride=FLAGS.stride,
    #                                                    batch_size=FLAGS.batch_size,
    #                                                    frac_data=FLAGS.frac_data,
    #                                                    label_f=etl.center_pixel)

    # resultspath = FLAGS.results_basepath  + FLAGS.experiment_name + ".json"
    # log = maybe_load_logfile(resultspath)
    # log['num_training_examples'] = num_examples
    # log['num_val_examples'] = len(xval)

    # json.dump(dict((k,v) for k,v in log.iteritems() if k in printable_params),\
    #           sys.stdout, indent=2)

    # xbatch, _ = next(train_iter())
    # example = xbatch[0]
    # ndim = example.shape[0]
    # num_channels = example.shape[-1]
    # xplaceholder = tf.placeholder(tf.float32, shape=(None, ndim, ndim, num_channels))
    # yplaceholder = tf.placeholder(tf.int64, shape=(None))
    # global_step, train_step, learning_rate, predictor, loss, accuracy = resnet.train_ops(xplaceholder,
    #                                                                                      yplaceholder,
    #                                                                                      FLAGS.architecture,
    #                                                                                      num_classes=4)
    # init = tf.initialize_all_variables()
    # saver = tf.train.Saver()

    # # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # with tf.Session() as sess:
    #     sess.run(init)

    #     if not maybe_restore_model(sess, saver):
    #         # See validation accuracy for the randomly initialized model.
    #         val_acc = run_validation(sess, accuracy, xplaceholder, yplaceholder,  xval, yval)
    #         log['val_accs'].append(str(val_acc))
    #         print("\n%s\t epoch: %d train accuracy (last batch): %f test_accuracy %f"\
    #               %(FLAGS.experiment_name, curr_epoch, train_acc, val_acc))

    #     num_steps_total = FLAGS.num_epochs * num_examples
    #     train_accs = []
    #     test_accs = []
    #     for xbatch, ybatch in train_iter():
    #         num_examples_seen = global_step.eval() * FLAGS.batch_size
    #         curr_epoch = int(num_examples_seen / num_examples)

    #         lr = next_learning_rate(0.1, curr_epoch)

    #         _, train_loss, train_acc = sess.run([train_step, loss, accuracy],
    #                                             feed_dict={xplaceholder: xbatch, yplaceholder: ybatch, learning_rate: lr})

    #         train_accs.append(str(train_acc))

    #         sys.stdout.write("\n iter: %d epoch: %d/%d learning rate: %f training accuracy: %f"\
    #                          % (global_step.eval(),
    #                             curr_epoch,
    #                             FLAGS.num_epochs,
    #                             lr,
    #                             train_acc)); sys.stdout.write('\r'); sys.stdout.flush()

    #         run_validation_period = 100
    #         if global_step.eval() % run_validation_period == 0:
    #             val_acc = run_validation(sess, accuracy, xplaceholder, yplaceholder,  xval, yval)
    #             log['val_accs'].append(str(val_acc))
    #             print("\n%s\t epoch: %d train accuracy (last batch): %f test_accuracy %f" %(FLAGS.experiment_name, curr_epoch, train_acc, val_acc))

    #         if global_step.eval() > num_steps_total:
    #             break

    #         savelog(resultspath, log)
        
    # #     for epoch_i in xrange(FLAGS.num_epochs):
    # #         # shuffle training data for each epoch
    # #         idx = np.array(list(range(len(xtrain))))
    # #         np.random.shuffle(idx)
    # #         xtrain = np.array(xtrain)[idx]
    # #         ytrain = np.array(ytrain)[idx]

    # #         # training
    # #         train_accs = []
    # #         for batch_i in xrange(0, num_examples, FLAGS.batch_size):
    # #             xbatch = xtrain[batch_i : batch_i + FLAGS.batch_size]
    # #             ybatch = ytrain[batch_i : batch_i + FLAGS.batch_size]

    # #             # TODO optimize, don't need to track acc if you are already tracking loss.
    # #             _, train_loss, train_acc = sess.run([train_step, loss, accuracy],
    # #                                              feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})

    # #             # stringify for JSON serialization
    # #             train_accs.append(str(train_acc))

    # #             print("epoch: %d/%d batch: %d training_accuracy=%f" \
    # #                   %(epoch_i+1, FLAGS.num_epochs, batch_i/FLAGS.batch_size, train_acc))

    # #         # save the model immediately
    # #         saver.save(sess, savepath)

    # #         # testing
    # #         test_accs = []
    # #         predictions = []
    # #         for batch_i in xrange(0, len(xtest), FLAGS.batch_size):
    # #             xbatch = xtest[batch_i : batch_i + FLAGS.batch_size]
    # #             ybatch = ytest[batch_i : batch_i + FLAGS.batch_size]

    # #             acc, preds = sess.run([accuracy, predictor], \
    # #                                   feed_dict={xplaceholder: xbatch, yplaceholder: ybatch})
                
    # #             test_accs.append(acc)
    # #             predictions.append(preds)

    # #         log['test_predictions'] = pickle.dumps(np.concatenate(predictions))
    # #         log['train_accs'].append(train_accs)
    # #         test_acc = sum(test_accs) / len(test_accs)
    # #         log['test_accs'].append(str(test_acc))
    # #         log['num_epochs_completed'] += 1
    # #         print("%s\t epoch: %d test_accuracy=%f" %(FLAGS.experiment_name, epoch_i+1, test_acc))

    # #         with open(resultspath, 'w+') as logfile:
    # #             try:
    # #                 json.dump(log, logfile, indent=4)
    # #             except TypeError, e:
    # #                 print(log)
    # #                 raise TypeError, e

if __name__ == '__main__':
    tf.app.run()

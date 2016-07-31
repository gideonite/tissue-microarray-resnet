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

prog_start = time.time()

flags.DEFINE_string('architecture', '41_layers', '')
flags.DEFINE_integer('batch_size', 128, 'Number of examples per GD batch')
flags.DEFINE_string('cache_basepath', '/mnt/data/output/', '')
flags.DEFINE_boolean('resume', False, 'Resume training or clobber the stuff and start over.')
flags.DEFINE_boolean('debug', False, 'Run in debug mode. (Skips test set evaluation).')
flags.DEFINE_string('experiment_name', 'experiment_' + str(prog_start), '')
flags.DEFINE_string('label_f', 'center_pixel_4labels', 'Which function to use for calculating the label for a patch')
flags.DEFINE_string('results_basepath', '/mnt/code/notebooks/results/', '')
flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
flags.DEFINE_integer('log_frequency', 100, 'How often to record the train and validation errors.')
flags.DEFINE_integer('num_epochs', 20, 'Number of times to go over the dataset')
flags.DEFINE_integer('num_gpus', 4, 'Number of GPUs to use for training and testing.')
TOWER_NAME = 'tower'
LOG_PATH = FLAGS.results_basepath  + FLAGS.experiment_name + ".json"

# TODO make more of these functions private

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_log(log):
    with open(LOG_PATH, 'w+') as logfile:
        try:
            log['time_elapsed'] = str(time.time() - prog_start)
            json.dump(log, logfile, indent=4)
        except TypeError, e:
            print(log)
            raise TypeError, e

def maybe_load_logfile(path=LOG_PATH):
    printable_params = set(['architecture', 'num_epochs', 'num_epochs_completed',\
                            'experiment_name', 'batch_size',\
                            'patch_size', 'stride'])

    if os.path.exists(path) and FLAGS.resume:
        with open(path) as f:
            log = json.load(f)
            json.dump(dict((k,v) for k,v in log.iteritems() if k in printable_params),\
                    sys.stdout, indent=2)

    else:
        print("creating new experiment from scratch in '" + path + "'")
        log = {'cmd': " ".join(sys.argv), # TODO, want to separate cmd line args from code to automatically restart experiments.
               'architecture': [g._asdict() for g in resnet.architectures[FLAGS.architecture]],
               'train_accs': [],
               'val_accs': [],
               'num_epochs': FLAGS.num_epochs,
               'num_epochs_completed': 0,
               'experiment_name': FLAGS.experiment_name,
               'batch_size': FLAGS.batch_size,
               'patch_size': FLAGS.patch_size,
               'stride': FLAGS.stride}

        save_log(log)

    return log

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

def maybe_restore_model(sess, saver):
    # TODO fix this savepath BS
    # TODO maybe delete this func.
    MODEL_SAVEPATH = mkdir(FLAGS.cache_basepath + '/' + FLAGS.experiment_name) \
                     + '/' + FLAGS.experiment_name + '.checkpoint'
    try:
        if not FLAGS.clobber:
            saver.restore(sess, MODEL_SAVEPATH)
            return True
    except ValueError:
        pass

    return False

def tower_loss(scope, xplaceholder, yplaceholder):
    
    net = resnet.inference(xplaceholder, FLAGS.architecture)

    final_layer_types = {'center_pixel_2labels': lambda net: resnet.fully_connected(net, outdim=2),
                         'center_pixel_4labels': lambda net: resnet.fully_connected(net, outdim=4)}
    logits = final_layer_types[FLAGS.label_f](net)
    
    _ = resnet.loss(logits, yplaceholder)
    
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),
                                               yplaceholder), tf.float32))

    tf.add_to_collection('accuracies', accuracy)

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

def log_accs(log, iter, train_acc, val_acc, duration):
    log['train_accs'].append((iter, str(train_acc)))
    log['val_accs'].append((iter, str(val_acc)))
    print('iter: %d train acc: %0.2f validation acc: %0.2f examples/sec: %0.2f'
          %(iter, train_acc, val_acc, FLAGS.batch_size / duration))
    save_log(log)
    return True

def learning_rate_schedule(iter, num_iterations):
    base = 0.1

    if 0 <= iter < 0.10 * num_iterations:
        return base
    elif 0.10 * num_iterations <= iter < 0.5 * num_iterations:
        return base * 0.1
    else: # 0.5 * num_iterations <= iter
        return base * 0.01

def resume(sess, saver):
    if FLAGS.resume:
        MODEL_SAVEPATH = mkdir(FLAGS.cache_basepath + '/' + FLAGS.experiment_name) \
                         + '/' + FLAGS.experiment_name + '.checkpoint'
        latest = tf.train.latest_checkpoint(MODEL_SAVEPATH)
        if not latest:
            print "No checkpoint to continue from in", MODEL_SAVEPATH
            sys.exit(1)
        print "resuming...", latest
        saver.restore(sess, latest)


def train():
    log = maybe_load_logfile()
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # TODO remove this
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        learning_rate = tf.placeholder(tf.float32, shape=[])

        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9)
        
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        num_channels = 3
        tower_grads = []
        placeholders = []
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' %(TOWER_NAME, i)) as scope:
                    xplaceholder = tf.placeholder(tf.float32, shape=(None, FLAGS.patch_size, FLAGS.patch_size, num_channels),
                                                  name='xplaceholder_gpu%d' % i)
                    yplaceholder = tf.placeholder(tf.int64, shape=(None),
                                                  name='yplaceholder_gpu%d' % i)
                    placeholders.append((xplaceholder, yplaceholder))
                    
                    loss = tower_loss(scope, xplaceholder, yplaceholder)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    tf.get_variable_scope().reuse_variables()

        grads = _average_grads(tower_grads)

        train_op = optimizer.apply_gradients(grads, global_step=global_step)

        total_accuracy = tf.add_n(tf.get_collection('accuracies')) / FLAGS.num_gpus

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))

        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess.run(init)

        resume(sess, saver)

        # TODO make sure that the loss is never NaN just like in the
        # cifar10 example. The accuracy doesn't help with that.

        label_functions = {
            'center_pixel_2labels': lambda patch: etl.collapse_classes(etl.center_pixel(patch)),
            'center_pixel_4labels': etl.center_pixel
        }

        num_examples, train_iter, xval, yval = etl.dataset(path=FLAGS.cache_basepath,
                                                           patch_size=FLAGS.patch_size,
                                                           stride=FLAGS.stride,
                                                           batch_size=FLAGS.batch_size / FLAGS.num_gpus,
                                                           frac_data=FLAGS.frac_data,
                                                           label_f=label_functions[FLAGS.label_f])

        it = train_iter()
        num_iterations = num_examples * FLAGS.num_epochs / FLAGS.batch_size
        for iter in range(num_iterations):
            lr = learning_rate_schedule(iter, num_iterations)
            feed_dict = {learning_rate: lr}
            for gpu_i in range(FLAGS.num_gpus):
                xbatch, ybatch = next(it)
                xplaceholder, yplaceholder = placeholders[gpu_i]
                feed_dict[xplaceholder] = xbatch
                feed_dict[yplaceholder] = ybatch

            start_time = time.time()
            _, train_acc = sess.run([train_op, total_accuracy], feed_dict=feed_dict)
            duration = time.time() - start_time

            if iter % FLAGS.log_frequency == 0:
                val_accs = []
                for i in range(0, len(xval), FLAGS.batch_size):
                    val_feed_dict = {}
                    for gpu_i in range(FLAGS.num_gpus):
                        xplaceholder, yplaceholder = placeholders[gpu_i]
                        val_feed_dict[xplaceholder] = xval[i+gpu_i:i+(FLAGS.batch_size / FLAGS.num_gpus)]
                        val_feed_dict[yplaceholder] = yval[i+gpu_i:i+(FLAGS.batch_size / FLAGS.num_gpus)]

                    val_accs.append(sess.run(total_accuracy, feed_dict=val_feed_dict))

                val_acc = sum(val_accs) / len(val_accs)
                print("learning rate: %f" % lr)
                log_accs(log, iter=iter, train_acc=train_acc, val_acc=val_acc, duration=duration)

                # TODO fix this savepath BS
                MODEL_SAVEPATH = mkdir(FLAGS.cache_basepath + '/' + FLAGS.experiment_name) \
                                 + '/' + FLAGS.experiment_name + '.checkpoint'
                print('saving..')
                saver.save(sess, MODEL_SAVEPATH, global_step=global_step)

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

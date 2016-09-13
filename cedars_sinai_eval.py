import json
import tensorflow as tf
import time
import sys
import numpy as np

import resnet
import cedars_sinai_train as train
# TODO delete cedars_sinai_etl
import cedars_sinai_etl2 as etl
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('list_of_samples', '/mnt/data/validation.txt', '')

MODEL_SAVEPATH = FLAGS.cache_basepath + '/' + FLAGS.experiment_name

LOG_PATH = FLAGS.results_basepath  + FLAGS.experiment_name + ".json"

with open(LOG_PATH) as logfile:
    log = json.load(logfile)

arch = resnet.architectures[log['cmd'].split()[log['cmd'].split().index('--architecture') + 1]] # TODO hack
patch_size = log['patch_size']

num_channels = 3
xplaceholder = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, num_channels), name='xplaceholder')
yplaceholder = tf.placeholder(tf.int64, shape=(None), name='yplaceholder')

net = resnet.inference(xplaceholder, arch)            # TODO make sure to save this in the log.
logits = train.final_layer_types[FLAGS.label_f](net)  # TODO make sure to save this in the log.
preds_op = tf.argmax(logits, 1)
global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
train.resume(sess, saver)

def chunks(l, n):
    '''
    Yield successive n-sized chunks from l.
    http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    '''
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def main(_):
    eval_set = etl.read_list_of_numbers_or_fail(FLAGS.list_of_samples)
    images = [etl.imread(num) for num in eval_set]

    l,w,h = images[0].shape
    for img_idx, img in enumerate(images): # ~ 4 sec. / img
        start = time.time()
        mask = []
        patches = etl._patches(img, patch_size, 1) 
        for batch in chunks(patches, FLAGS.batch_size):
            preds = sess.run(preds_op, feed_dict={xplaceholder: batch})
            mask.extend(preds)

        duration = (time.time() - start) / 60
        print('sample num %d duration %0.2f min' %(eval_set[img_idx], duration))
        mask = np.array(mask).reshape([l-patch_size+1, w-patch_size+1])
        # cv2.copyMakeBorder(mask,
        #                    patch_size/2-1,
        #                    patch_size/2,
        #                    patch_size/2-1,
        #                    patch_size/2,
        #                    cv2.BORDER_CONSTANT,
        #                    value=[0, 0, 0]).shape

        np.save(MODEL_SAVEPATH + '/' + 'test' + str(eval_set[img_idx]) + '_preds.npy', mask)

if __name__ == '__main__':
    tf.app.run()

import json
import tensorflow as tf
import time
import sys
import numpy as np

import resnet
import cedars_sinai_train as train
import cedars_sinai_etl2 as etl
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('list_of_samples', '/mnt/data/validation.txt', '')

MODEL_SAVEPATH = FLAGS.cache_basepath + '/' + FLAGS.experiment_name

LOG_PATH = FLAGS.results_basepath  + FLAGS.experiment_name + ".json"

with open(LOG_PATH) as logfile:
    log = json.load(logfile)
import pdb; pdb.set_trace

num_channels = 3
xplaceholder = tf.placeholder(tf.float32, shape=(None, FLAGS.patch_size, FLAGS.patch_size, num_channels), name='xplaceholder')
yplaceholder = tf.placeholder(tf.int64, shape=(None), name='yplaceholder')

arch = resnet.architectures[log['cmd'].split()[log['cmd'].split().index('--architecture') + 1]] # TODO hack
net = resnet.inference(xplaceholder, arch)            # TODO make sure to save this in the log.
logits = train.final_layer_types[FLAGS.label_f](net)  # TODO make sure to save this in the log.
preds_op = tf.argmax(logits, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess.run(init)
train.resume(sess, saver)

def _batchify(img):
   patches = etl._patches(img, FLAGS.patch_size, 1) 

   batch = []
   for patch in patches:
       batch.append(patch)
       if len(batch) >= FLAGS.batch_size:
           yield batch
           batch = []
   yield batch # flush

def main(_):
    eval_set = etl.read_list_of_numbers_or_fail(FLAGS.list_of_samples)
    images = [etl.imread(num) for num in eval_set]

    l,w,h = images[0].shape
    for img_idx, img in enumerate(images): # ~ 4 sec. / img
        start = time.time()
        # mask = np.empty([l-FLAGS.patch_size*2, w-FLAGS.patch_size*2, 1])
        mask = []
        patches = etl._patches(img, FLAGS.patch_size, 1) 
        for batch in _batchify(img):
            preds = sess.run(preds_op, feed_dict={xplaceholder: batch})
            mask.extend(preds)

        duration = (time.time() - start) / 60
        print('sample num %d duration %0.2f min' %(eval_set[img_idx], duration))
        mask = np.array(mask).reshape([l-FLAGS.patch_size+1, w-FLAGS.patch_size+1])
        mask = cv2.copyMakeBorder(mask,
                                  FLAGS.patch_size/2-1,
                                  FLAGS.patch_size,
                                  FLAGS.patch_size-1,
                                  FLAGS.patch_size,
                                  cv2.BORDER_CONSTANT,value=[0,0,0])

        np.save(MODEL_SAVEPATH + '/' + 'test' + str(eval_set[img_idx]) + '_preds.npy', mask)

    # create the model based on what's in the log file.
    # for each test file,
        # for each pixel in test file
            # evaluate the patch at that pixel and save in a matrix
    # pad the matrix to match dimensions of the image (dim mismatch
    # because of patch size). Basically need to patch with patchsize
    # all around the image.

    # save as .png in the MODEL_SAVEPATH folder as filename_pred.png

if __name__ == '__main__':
    tf.app.run()

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# from input_data import *
import input_data
import c3d_model
import numpy as np

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
FLAGS = flags.FLAGS


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           c3d_model.NUM_FRAMES_PER_CLIP,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder


def _variable_on_cpu(name, shape, initializer):
    # with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def run_test():
    model_name = "./sports1m_finetuning_ucf101.model"
    test_list_file = './list/test1.list'
    num_test_videos = len(list(open(test_list_file, 'r')))
    print("Number of test videos={}".format(num_test_videos))

    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
    with tf.variable_scope('var_name') as var_scope:
        weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
            'deconv1': _variable_with_weight_decay('deconv1', [1, 4, 4, 1, 512], 0.0005),
            'deconv2': _variable_with_weight_decay('deconv2', [1, 3, 3, 1, 1], 0.0005)
        }
        biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0)
        }
    logits = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            logit = c3d_model.inference_c3d(
                images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :, :], 0.6,
                FLAGS.batch_size, weights, biases)
            logits.append(logit)
    logits = tf.concat(0, logits)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    init = tf.initialize_all_variables()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)
    # And then after everything is built, start the training loop.
    next_start_pos = 0
    all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
    for step in xrange(all_steps):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                test_list_file,
                FLAGS.batch_size * gpu_num,
                start_pos=next_start_pos
            )
        predict_data = sess.run(logits, feed_dict={images_placeholder: test_images})
        return predict_data
    print("done")


def main(_):
    run_test()


# run_test()
if __name__ == '__main__':
    tf.app.run()


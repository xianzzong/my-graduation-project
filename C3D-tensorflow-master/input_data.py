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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
# import cv2
import time
from scipy.misc import imresize


def get_frames_data(filename, num_frames_per_clip=16):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    ret_arr = []
    s_index = 0
    for parent, dirnames, filenames in os.walk(filename):
        if (len(filenames) < num_frames_per_clip):
            return [], s_index
        filenames = sorted(filenames)
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
        for i in range(s_index, s_index + num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr, s_index


def read_images_from_dir(tmp_data, num_frames_per_clip=16, crop_size=112, flag=True):
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    img_datas = []
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if (img.width > img.height):
            scale = float(crop_size) / float(img.height)
            img = np.array(imresize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(imresize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        if flag:
            img = img[int((img.shape[0] - crop_size) // 2):int((img.shape[0] - crop_size) // 2 + crop_size),
                  int((img.shape[1] - crop_size) // 2):int((img.shape[1] - crop_size) // 2 + crop_size), :] - np_mean[j]
        else:
            img = img[int((img.shape[0] - crop_size) // 2):int((img.shape[0] - crop_size) // 2 + crop_size),
                  int((img.shape[1] - crop_size) // 2):int((img.shape[1] - crop_size) // 2 + crop_size), :]
        img_datas.append(img)
    return img_datas


def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
    lines = open(filename, 'r')
    read_dirnames = []
    data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
    np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
    # Forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True
    if shuffle:
        video_indices = range(len(lines))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        # Process videos sequentially
        video_indices = range(start_pos, len(lines))
    for index in video_indices:
        if (batch_index >= batch_size):
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname_data = line[0]
        dirname_label = line[1]
        if not shuffle:
            print("Loading a video clip from {}...".format(dirname_data))
        tmp_data, _ = get_frames_data(dirname_data, num_frames_per_clip)
        tmp_label, _ = get_frames_data(dirname_label, num_frames_per_clip)
        img_datas = []
        img_labels = []
        if (len(tmp_data) != 0):
            img_datas = read_images_from_dir(tmp_data)
            img_labels = read_images_from_dir(tmp_label, flag=False)
            data.append(img_datas)
            label.append(img_labels)
            batch_index = batch_index + 1
            read_dirnames.append(dirname_data)

    # pad (duplicate) data/label if less than batch_size
    valid_len = len(data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            data.append(img_datas)
            label.append(img_labels)

    np_arr_data = np.array(data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.float32)

    return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len


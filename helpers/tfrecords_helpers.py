# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# MODIFIED FROM ORIGINAL

import os
import tensorflow as tf
import numpy as np
from scipy import misc

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_list_to_tfrecords(path_list, label_list, tfrecords_dir, tfrecords_name):

    """Converts a dataset to grayscale tfrecords."""

    num_files = len(path_list)
    num_labels = len(label_list)
    if num_files != num_labels:
        raise ValueError('Images size %d does not match label size %d.' %
                     (num_files, num_labels))

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)
    fullpath = os.path.join(tfrecords_dir, tfrecords_name)
    print('Writing to ', fullpath)
    writer = tf.python_io.TFRecordWriter(fullpath)

    for index in range(num_files):
        image = misc.imread(path_list[index])
        image = np.uint8(np.mean(image, axis=2)) # CONVERT TO GRAYSCALE
        image_str = image.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(int(label_list[index])),
            "image_raw": _bytes_feature(image_str)}))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename_queue, batch_size, input_size):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "label": tf.FixedLenFeature([], tf.int64),
        "image_raw": tf.FixedLenFeature([], tf.string)
    })

    # height = tf.cast(features['height'], tf.int32)
    # width = tf.cast(features['width'], tf.int32)
    # depth = tf.cast(features['depth'], tf.int32)

    label = tf.cast(features["label"], tf.int32)
    image_raw = tf.decode_raw(features["image_raw"], tf.uint8)
    image_raw = tf.cast(tf.reshape(image_raw, tf.stack(input_size)), tf.float32)

    # images, labels = tf.train.shuffle_batch([image_raw, label], batch_size=batch_size, capacity=30, num_threads=4, min_after_dequeue=10)
    images, labels = tf.train.batch([image_raw, label], batch_size, capacity=32, num_threads=1)

    return images, labels

#######################################3333333
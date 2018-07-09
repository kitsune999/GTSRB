from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
from data import cifarnet_preprocessing as impro

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


def read_and_decode(filename_queue, mode):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            # 'height': tf.FixedLenFeature([], tf.int64),
            # 'width': tf.FixedLenFeature([], tf.int64),
            # 'depth': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    # height = tf.cast(features['height'], tf.int32)
    # width = tf.cast(features['width'], tf.int32)
    # depth = tf.cast(features['depth'], tf.int32)
    return image, label


def build_input(data_path, image_size, num_classes, batch_size, mode, eval_once=False, val_once=False):
    raw_image_width = 48
    raw_image_height = 48
    depth = 3

    if image_size < 0:
        image_size = raw_image_width

    if mode == 'train':
        filename = TRAIN_FILE
    elif mode == 'eval':
        filename = TEST_FILE
    else:
        filename = VALIDATION_FILE

    filename = os.path.join(data_path,filename)

    if mode == 'eval' and eval_once:
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    elif mode == 'eval' and val_once:
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
    else:
        filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue, mode)
    image = tf.reshape(image, [raw_image_width, raw_image_height, depth])
    image.set_shape([raw_image_width, raw_image_height, depth])

    image = tf.cast(image, tf.float32)

    label = tf.reshape(label, [1])

    if mode == 'train':
        image = impro.preprocess_image(image, image_size, image_size, True)

        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 16
    elif mode == 'eval':
        image = impro.preprocess_image(image, image_size, image_size, False)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1
    else:
        image = impro.preprocess_image(image, image_size, image_size, False)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, depth], [1]])
        num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)

    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    return images, labels

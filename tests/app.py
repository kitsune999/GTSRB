import tensorflow as tf
import models
import cv2
import numpy as np
import sys
import argparse
from matplotlib import pyplot as plt

FLAGS = None


def read_image(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, (48, 48))
    return image


def main(unuse_argv):
    h_params = models.get_model_HParams('IslingST')
    hps = h_params(batch_size=1,
                   num_classes=43,
                   min_lrn_rate=0.0001,
                   lrn_rate=0.1,
                   optimizer='sgd',
                   weight_decay_rate=0.0005,
                   dropout=0)

    labels = tf.placeholder(tf.float32, [1, 43])

    image = read_image(FLAGS.image)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        images = tf.reshape(image, [1, 48, 48, 3])

        model = models.get_model('IslingST', hps, images, labels, 'val')
        model.build_graph()

        sess.run(init)
        saver = tf.train.Saver()

        try:
            ckpt_state = tf.train.get_checkpoint_state('../results/IslingST')
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', '../results/IslingST')
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        predictions = tf.argmax(model.predictions, axis=1)

        label = sess.run(predictions)

        print label
        #
        # image = model.st_images[0].eval()
        #
        # plt.imshow(image)
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        type=str,
        default='',
        help='Image url'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

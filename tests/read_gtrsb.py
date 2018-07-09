import tensorflow as tf
import numpy as np
from PIL import Image
import GTRSB_input
import cv2

data_path = './data'

# image, label = GTRSB_input.test('train')

with tf.Session() as sess:
    # Start populating the filename queue.
    images, labels = GTRSB_input.build_input(data_path, 128, 'train')
    # images, labels = GTRSB_input.inputs('train', 128)
    images = tf.cast(images, tf.uint8)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1):  # length of your filename list
        images, labels = sess.run([images, labels])
        image = images[i]  # here is your image Tensor :)
        label = labels[i]

        img = Image.fromarray(image, 'RGB')
        img.save("./" + str(i) + '-train.png')

        print (label)
        # print(image)
        # print(label)

        # cv2.imshow('aaa', image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    coord.request_stop()
    coord.join(threads)

from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages
from spatial_transformer import transformer
from tf_utils import weight_variable, bias_variable, dense_to_one_hot

slim = tf.contrib.slim

IslingST_HParams = namedtuple('HParams',
                              'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                              'optimizer, weight_decay_rate, dropout')


class IslingST(object):
    def __init__(self, hps, images, labels, mode):
        """Vgg constructor.
    
        Args:
          hps: Hyperparameters.
          images: Batches of images. [batch_size, image_size, image_size, 3]
          labels: Batches of labels. [batch_size, num_classes]
          mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.labels = labels
        self.mode = mode
        self._dropout = hps.dropout

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        """Build the core model within the graph."""
        # with tf.variable_scope('ST_1'):
        #     x = self._images
        #     x = self._spatial_transformer('ST_1', x, 3, [200, 300, 30])

        with tf.variable_scope('conv_1_0'):
            x = self._images
            x = self._conv_Bn_ReLU('conv_1_0', x, 3, 3, 64, self._stride_arr(1))
        with tf.variable_scope('conv_1_1'):
            x = self._conv_Bn_ReLU('conv_1_1', x, 1, 64, 32, self._stride_arr(1))
        with tf.variable_scope('conv_1_2'):
            x = self._conv_Bn_ReLU('conv_1_2', x, 3, 32, 32, self._stride_arr(1))
        with tf.variable_scope('conv_1_3'):
            x = self._conv_Bn_ReLU('conv_1_3', x, 1, 32, 64, self._stride_arr(1))
        with tf.variable_scope('pool_1'):
            x = self._max_pool(x, 'pool_1')

        # with tf.variable_scope('ST_2'):
        #     x = self._spatial_transformer('ST_2', x, 64, [150, 150, 20])

        with tf.variable_scope('conv_2_1'):
            x = self._conv_Bn_ReLU('conv_2_1', x, 1, 64, 64, self._stride_arr(1))
        with tf.variable_scope('conv_2_2'):
            x = self._conv_Bn_ReLU('conv_2_2', x, 3, 64, 64, self._stride_arr(1))
        with tf.variable_scope('conv_2_3'):
            x = self._conv_Bn_ReLU('conv_2_3', x, 1, 64, 128, self._stride_arr(1))
        with tf.variable_scope('pool_2'):
            x = self._max_pool(x, 'pool_2')

        with tf.variable_scope('conv_3_1'):
            x = self._conv_Bn_ReLU('conv_3_1', x, 1, 128, 128, self._stride_arr(1))
        with tf.variable_scope('conv_3_2'):
            x = self._conv_Bn_ReLU('conv_3_2', x, 3, 128, 128, self._stride_arr(1))
        with tf.variable_scope('conv_3_3'):
            x = self._conv_Bn_ReLU('conv_3_3', x, 1, 128, 256, self._stride_arr(1))
        with tf.variable_scope('pool_3'):
            x = self._max_pool(x, 'pool_3')

        with tf.variable_scope('fc_1'):
            x = self._fully_connected('fc_1', x, 512)
            x = self._batch_norm2('fc_1', x)
            x = self._relu(x)

        x = slim.dropout(x, self._dropout,
                         is_training=(self.mode == 'train' and self._dropout > 0),
                         scope='dropout1')
        # with tf.variable_scope('fc_2'):
        #     x = self._fully_connected('fc_2', x, 384)
        #     x = self._batch_norm2('fc_2', x)
        #     x = self._relu(x)
        # if self.mode == 'train':
        #     tf.nn.dropout(x, self._dropout)

        with tf.variable_scope('logit'):
            logits = self._fully_connected('logit', x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()
            tf.summary.scalar('cost', self.cost)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        else:
            optimizer = tf.train.AdamOptimizer(self.lrn_rate)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _conv_Bn_ReLU(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = in_filters * in_filters * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            b = tf.get_variable('biases', [out_filters],
                                initializer=tf.constant_initializer())
            x = tf.nn.conv2d(x, kernel, strides, padding='SAME')
            x = tf.nn.bias_add(x, b)
            x = self._batch_norm(name, x)
            tf.summary.histogram('pre_activations', x)
            x = self._relu(x)
            tf.summary.histogram('activations', x)
            return x

    def _conv_ReLU(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = in_filters * in_filters * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            b = tf.get_variable('biases', [out_filters],
                                initializer=tf.constant_initializer())
            x = tf.nn.conv2d(x, kernel, strides, padding='SAME')
            x = tf.nn.bias_add(x, b)
            x = self._batch_norm(name, x)
            tf.summary.histogram('pre_activations', x)
            x = self._relu(x)
            tf.summary.histogram('activations', x)
            return x

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, name, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        with tf.variable_scope(name):
            w = tf.get_variable(
                'DW', [x.get_shape()[1], out_dim], tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer())
            result = tf.nn.xw_plus_b(x, w, b)
            tf.summary.histogram('fully_connected_result', result)
            return result

    def _max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _frac_max_pool(selfself, x, name):
        p, _, _ = tf.nn.fractional_max_pool(x, pooling_ratio=[1.0, 1.3, 1.3, 1.0], name=name)
        return p

    # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2, 3], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _batch_norm2(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _spatial_transformer(self, name, x, in_filters, arr_out_filters):
        width = x.get_shape().as_list()[1]
        height = x.get_shape().as_list()[2]

        with tf.variable_scope(name):
            W_fc_loc1 = weight_variable([width * height * in_filters, arr_out_filters[2]])
            b_fc_loc1 = bias_variable([arr_out_filters[2]])

            W_fc_loc2 = weight_variable([arr_out_filters[2], 6])
            # Use identity transformation as starting point
            initial = np.array([[1., 0, 0], [0, 1., 0]])
            initial = initial.astype('float32')
            initial = initial.flatten()
            b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

            x_reshape = tf.reshape(x, [self.hps.batch_size, -1])
            # %% Define the two layer localisation network
            h_fc_loc1 = tf.nn.tanh(tf.matmul(x_reshape, W_fc_loc1) + b_fc_loc1)
            h_fc_loc1 = self._batch_norm2(name, h_fc_loc1)
            h_fc_loc1 = self._relu(h_fc_loc1)

            h_fc_loc1 = slim.dropout(h_fc_loc1, self._dropout,
                                     is_training=(self.mode == 'train'
                                                  and self._dropout
                                                  and self._dropout > 0),
                                     scope='dropout')

            h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2)

            # %% We'll create a spatial transformer module to identify discriminative
            # %% patches
            out_size = (width, height)
            h_trans = transformer(x, h_fc_loc2, out_size)
            h_trans = tf.reshape(h_trans, [self.hps.batch_size, width, height, in_filters])
        return h_trans

# Copyright 2016 Kitsune. All Rights Reserved.
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
import time
import six
import sys

import numpy as np
import tensorflow as tf

from data import GTRSB_input
import models
# import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'Isling', 'model name.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_integer('image_size', -1, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', './results/' + FLAGS.model + '/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', './results/' + FLAGS.model + '/eval',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 106,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', True,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('val_dir', './results/' + FLAGS.model + '/val',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('val_batch_count', 40,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('val_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', './results/' + FLAGS.model,
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')

data_path = './data'


def train(hps):
    """Training loop."""
    images, labels = GTRSB_input.build_input(data_path, FLAGS.image_size,
                                             hps.num_classes, hps.batch_size, FLAGS.mode)

    model = models.get_model(FLAGS.model, hps, images, labels, FLAGS.mode)
    model.build_graph()

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=FLAGS.train_dir,
        summary_op=tf.summary.merge([model.summaries,
                                     tf.summary.scalar('Precision', precision)]))

    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=FLAGS.log_root,
        save_steps=200,
        saver=tf.train.Saver())

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = hps.lrn_rate

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            if train_step < 10000:
                self._lrn_rate = 0.1
            elif train_step < 20000:
                self._lrn_rate = 0.01
            elif train_step < 60000:
                self._lrn_rate = 0.001
            else:
                self._lrn_rate = 0.0001

    # list_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_1_0')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_1_1')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_1_2')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_1_3')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pool_1')
    #
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_2_1')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_2_2')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_2_3')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pool_2')
    #
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_3_1')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_3_2')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_3_3')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pool_3')
    #
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc_1')
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='logit')
    # # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gradients')
    #
    # list_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tensorflow')
    #
    # saver = tf.train.Saver(list_var)

    with tf.train.MonitoredTrainingSession(
            # is_chief=True,
            checkpoint_dir=FLAGS.log_root,
            hooks=[logging_hook, _LearningRateSetterHook(), saver_hook],
            chief_only_hooks=[summary_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            save_checkpoint_secs=0,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        # ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        # saver.restore(mon_sess, ckpt_state.model_checkpoint_path)
        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)


def evaluate(hps):
    """Eval loop."""
    images, labels = GTRSB_input.build_input(data_path, FLAGS.image_size,
                                             hps.num_classes, hps.batch_size, FLAGS.mode)
    model = models.get_model(FLAGS.model, hps, images, labels, FLAGS.mode)
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [model.summaries, model.cost, model.predictions,
                 model.labels, model.global_step])

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
            break

        time.sleep(1)


def validation(hps):
    images, labels = GTRSB_input.build_input(data_path, FLAGS.image_size,
                                             hps.num_classes, hps.batch_size, FLAGS.mode)
    model = models.get_model(FLAGS.model, hps, images, labels, FLAGS.mode)
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.val_dir)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    best_precision = 0.0

    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to validate yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [model.summaries, model.cost, model.predictions,
                 model.labels, model.global_step])

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.val_once:
            break

        time.sleep(1)


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    if FLAGS.mode == 'train':
        batch_size = 128
    elif FLAGS.mode == 'eval':
        batch_size = 120
    else:
        batch_size = 100

    num_classes = 43

    h_params = models.get_model_HParams(FLAGS.model)
    hps = h_params(batch_size=batch_size,
                   num_classes=num_classes,
                   min_lrn_rate=0.0001,
                   lrn_rate=0.1,
                   optimizer='mom',
                   weight_decay_rate=0.0005,
                   dropout=0.3)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode == 'eval':
            evaluate(hps)
        else:
            validation(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

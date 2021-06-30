# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train a ResNet-50 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED.mlperf.submissions.training.v1_0.models.mlp_log import mlp_log
from REDACTED.mlperf.submissions.training.v1_0.models.resnet import imagenet_input
from REDACTED.mlperf.submissions.training.v1_0.models.resnet import lars_util
from REDACTED.mlperf.submissions.training.v1_0.models.resnet import resnet_model
from REDACTED.mlperf.submissions.training.v1_0.models.util import train_and_eval_runner


FLAGS = flags.FLAGS


# Model specific flags
flags.DEFINE_string(
    'data_dir', default=None,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_integer(
    'resnet_depth', default=50,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_integer(
    'train_steps', default=112590,
    help=('The number of steps to use for training. Default is 112590 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'num_label_classes', default=1000, help='Number of classes, at least 2')

flags.DEFINE_integer(
    'steps_per_eval', default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'dataset_threadpool_size', default=48,
    help=('The size of the private datapool size in dataset.'))

flags.DEFINE_integer('num_replicas', default=8, help=('Number of replicas.'))

flags.DEFINE_string(
    'precision', default='bfloat16',
    help=('Precision to use; one of: {bfloat16, float32}'))

flags.DEFINE_float(
    'base_learning_rate', default=0.1,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'weight_decay', default=1e-4,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.0,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_bool('enable_lars',
                  default=False,
                  help=('Enable LARS optimizer for large batch training.'))

flags.DEFINE_float(
    'stop_threshold', default=0.759, help=('Stop threshold for MLPerf.'))

flags.DEFINE_integer('image_size', 224, 'The input image size.')

flags.DEFINE_integer(
    'distributed_group_size',
    default=1,
    help=('When set to > 1, it will enable distributed batch normalization'))

flags.DEFINE_integer(
    'logical_devices',
    default=2,
    help=('Logical number of accelerators per chip'))

flags.DEFINE_integer(
    'tpu_topology_dim_count',
    default=2,
    help=('Number of dimensions in interconnect topology'))

tf.flags.DEFINE_multi_integer(
    'input_partition_dims',
    default=None,
    help=('Number of partitions on each dimension of the input. Each TPU core'
          ' processes a partition of the input image in parallel using spatial'
          ' partitioning.'))

flags.DEFINE_bool(
    'use_space_to_depth',
    default=False,
    help=('Enable space-to-depth optimization for conv-0.'))

flags.DEFINE_integer(
    'bfloat16_replica_threshold',
    default=128,
    help=('Threshold for enabling bfloat16 for cross replica sum.'))

flags.DEFINE_integer(
    'num_experiments', default=1, help=('Number of bechmark expriments to run'))

# Learning rate schedule
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

  decay_rate = (scaled_lr * LR_SCHEDULE[0][0] *
                current_epoch / LR_SCHEDULE[0][1])
  for mult, start_epoch in LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def resnet_model_fn(features, labels, is_training):
  """The model_fn for ResNet to be used with TPU.

  Args:
    features: `Tensor` of batched images.
    labels: `Tensor` of labels for the data samples
    is_training: whether this is training

  Returns:
    train_op, logits
  """
  if isinstance(features, dict):
    features = features['feature']

  if FLAGS.use_space_to_depth:
    if FLAGS.train_batch_size // FLAGS.num_replicas > 8:
      features = tf.reshape(
          features, [FLAGS.image_size // 2, FLAGS.image_size // 2, 12, -1])
      features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC
    else:
      features = tf.reshape(
          features, [FLAGS.image_size // 2, FLAGS.image_size // 2, -1, 12])
      features = tf.transpose(features, [2, 0, 1, 3])  # HWNC to NHWC
  else:
    if FLAGS.train_batch_size // FLAGS.num_replicas > 8:
      features = tf.reshape(features,
                            [FLAGS.image_size, FLAGS.image_size, 3, -1])
      features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC
    else:
      features = tf.reshape(features,
                            [FLAGS.image_size, FLAGS.image_size, -1, 3])
      features = tf.transpose(features, [2, 0, 1, 3])  # HWCN to NHWC

  # Normalize the image to zero mean and unit variance.
  if FLAGS.use_space_to_depth:
    features -= tf.constant(MEAN_RGB, shape=[1, 1, 12], dtype=features.dtype)
    features /= tf.constant(STDDEV_RGB, shape=[1, 1, 12], dtype=features.dtype)
  else:
    features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
    features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)

  # This nested function allows us to avoid duplicating the logic which
  # builds the network, for different values of --precision.
  def build_network():
    with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
      network = resnet_model.resnet_v1(
          resnet_depth=FLAGS.resnet_depth,
          num_classes=FLAGS.num_label_classes,
          use_space_to_depth=FLAGS.use_space_to_depth,
          num_replicas=FLAGS.num_replicas,
          distributed_group_size=FLAGS.distributed_group_size,
          logical_devices=FLAGS.logical_devices,
          tpu_topology_dim_count=FLAGS.tpu_topology_dim_count)
      return network(inputs=features, is_training=is_training)

  if FLAGS.precision == 'bfloat16':
    with tf.tpu.bfloat16_scope():
      logits = build_network()
    logits = tf.cast(logits, tf.float32)
  elif FLAGS.precision == 'float32':
    logits = build_network()

  if not is_training:
    total_correct = tf.reduce_sum(
        tf.cast(
            tf.equal(tf.cast(tf.argmax(logits, axis=1), labels.dtype), labels),
            tf.int32))
    return None, {'total_correct': tf.reshape(total_correct, [-1])}

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  one_hot_labels = tf.one_hot(labels, FLAGS.num_label_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      onehot_labels=one_hot_labels,
      label_smoothing=FLAGS.label_smoothing)

  # Add weight decay to the loss for non-batch-normalization variables.
  if FLAGS.enable_lars:
    loss = cross_entropy
  else:
    loss = cross_entropy + FLAGS.weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name
    ])

  global_step = tf.train.get_or_create_global_step()
  steps_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
  current_epoch = (tf.cast(global_step, tf.float32) / steps_per_epoch)

  if FLAGS.enable_lars:
    learning_rate = 0.0
    optimizer = lars_util.init_lars_optimizer(current_epoch)
  else:
    learning_rate = learning_rate_schedule(current_epoch)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=FLAGS.momentum, use_nesterov=True)

  use_bf16_allreduce = (FLAGS.num_replicas <= FLAGS.bfloat16_replica_threshold)
  optimizer = optimizer if use_bf16_allreduce else tf.tpu.CrossShardOptimizer(
      optimizer)

  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    if use_bf16_allreduce:
      scaled_loss = loss / FLAGS.num_replicas
      tvars = tf.trainable_variables()
      grads = tf.gradients(scaled_loss, tvars)
      grads_tvars = zip(grads, tvars)
      grads_tvars = [(tf.cast(
          tf.tpu.cross_replica_sum(tf.cast(g, tf.bfloat16)), tf.float32), v)
                     for g, v in grads_tvars]
      train_op = optimizer.apply_gradients(grads_tvars, global_step=global_step)
    else:
      train_op = optimizer.minimize(loss, global_step)
  return train_op, None


def run_benchmark():
  """Execute a benchmark iteration."""

  def eval_init_fn(cur_step):
    """Executed before every eval."""
    steps_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
    epoch = cur_step // steps_per_epoch
    mlp_log.mlperf_print(
        'block_start',
        None,
        metadata={
            'first_epoch_num': epoch,
            'epoch_count': 4
        })

  def eval_finish_fn(cur_step, eval_output, summary_writer):
    """Executed after every eval."""
    steps_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
    epoch = cur_step // steps_per_epoch
    eval_accuracy = float(np.sum(
        eval_output['total_correct'])) / FLAGS.num_eval_images

    if summary_writer:
      with tf.Graph().as_default():
        summary_writer.add_summary(
            tf.Summary(value=[
                tf.Summary.Value(tag='accuracy', simple_value=eval_accuracy)
            ]), cur_step)
    mlp_log.mlperf_print(
        'eval_accuracy',
        eval_accuracy,
        metadata={
            'epoch_num': epoch + FLAGS.iterations_per_loop // steps_per_epoch
        })
    mlp_log.mlperf_print(
        'block_stop',
        None,
        metadata={
            'first_epoch_num': epoch,
            'epoch_count': 4
        })
    if eval_accuracy >= FLAGS.stop_threshold:
      mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
      return True
    else:
      return False

  def run_finish_fn(success):
    if not success:
      mlp_log.mlperf_print('run_stop', None, metadata={'status': 'abort'})
    mlp_log.mlperf_print('run_final', None)

  low_level_runner = train_and_eval_runner.TrainAndEvalRunner(
      FLAGS.iterations_per_loop, FLAGS.train_steps,
      int(math.ceil(FLAGS.num_eval_images / FLAGS.eval_batch_size)),
      FLAGS.num_replicas)

  mlp_log.mlperf_print('cache_clear', True)
  mlp_log.mlperf_print('init_start', None)
  mlp_log.mlperf_print('global_batch_size', FLAGS.train_batch_size)
  mlp_log.mlperf_print('lars_opt_weight_decay', FLAGS.weight_decay)
  mlp_log.mlperf_print('lars_opt_momentum', FLAGS.momentum)
  mlp_log.mlperf_print('lars_epsilon', FLAGS.lars_epsilon)
  mlp_log.mlperf_print('submission_benchmark', 'resnet')
  mlp_log.mlperf_print('submission_division', 'closed')
  mlp_log.mlperf_print('submission_org', 'google')
  mlp_log.mlperf_print('submission_platform', 'tpu-v3-%d' % FLAGS.num_replicas)
  mlp_log.mlperf_print('submission_status', 'research')
  mlp_log.mlperf_print(
      'model_bn_span',
      FLAGS.distributed_group_size *
      (FLAGS.train_batch_size // FLAGS.num_replicas))
  if FLAGS.enable_lars:
    mlp_log.mlperf_print('opt_name', 'lars')
  else:
    mlp_log.mlperf_print('opt_name', 'sgd')

  mlp_log.mlperf_print('lars_opt_learning_rate_decay_poly_power', 2.0)
  lars_hp = lars_util.compute_lars_hp()
  plr = lars_hp['base_lr']
  w_epochs = lars_hp['w_epochs']
  end_lr = lars_hp['end_lr']
  w_steps = lars_hp['w_steps']
  mlp_log.mlperf_print('lars_opt_base_learning_rate', plr)
  mlp_log.mlperf_print('lars_opt_learning_rate_warmup_epochs', w_epochs)
  mlp_log.mlperf_print('lars_opt_end_learning_rate', end_lr)
  mlp_log.mlperf_print('lars_opt_learning_rate_decay_steps',
                       FLAGS.train_steps - w_steps + 1)

  assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', (
      'Invalid value for --precision flag; must be bfloat16 or float32.')
  input_dtype = tf.bfloat16 if FLAGS.precision == 'bfloat16' else tf.float32
  cache_decoded_image = True if FLAGS.num_replicas >= 512 else False
  imagenet_train, imagenet_eval = [
      imagenet_input.get_input_fn(  # pylint: disable=g-complex-comprehension
          FLAGS.data_dir,
          is_training,
          input_dtype,
          FLAGS.image_size,
          FLAGS.input_partition_dims is None,
          cache_decoded_image=cache_decoded_image)
      for is_training in [True, False]
  ]

  low_level_runner.initialize(imagenet_train, imagenet_eval, resnet_model_fn,
                              FLAGS.train_batch_size, FLAGS.eval_batch_size,
                              FLAGS.input_partition_dims)

  mlp_log.mlperf_print('train_samples', FLAGS.num_train_images)
  mlp_log.mlperf_print('eval_samples', FLAGS.num_eval_images)
  mlp_log.mlperf_print('init_stop', None)
  mlp_log.mlperf_print('run_start', None)
  low_level_runner.train_and_eval(eval_init_fn, eval_finish_fn, run_finish_fn)


def main(unused_argv):
  for i in range(0, FLAGS.num_experiments):
    tf.logging.info('Start Iteration: %d' % i)
    run_benchmark()
    tf.Session.reset(FLAGS.master)


if __name__ == '__main__':
  app.run(main)

# Lint as: python3
"""Unet3d main training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

import math
import time

from absl import app
from absl import flags
from absl import logging

import jax
from jax import config
import jax.numpy as jnp
from jax.util import partial

import numpy as onp
import tensorflow.compat.v2 as tf

import REDACTED.learning.deepmind.REDACTED.client.google as xm
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.data_loading import data_loader
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.data_loading import input_reader
# Import below lines so that we do not have flag errors when reusing the same
# REDACTED file.
# pylint: disable=unused-import
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.runtime import arguments
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.runtime import inference
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.runtime import training
# pylint: enable=unused-import

flags.DEFINE_string(
    'init_dummy_file',
    default=None,
    help='Read a dummy file to initialize datacenter connection.')

flags.DEFINE_bool(
    'space_filling_device_assignment', default=False,
    help='Make device assignment with space filling curves.')


flags.DEFINE_bool(
    'hardware_rng', default=True,
    help='Enable faster RNG generation.')

flags.DEFINE_bool(
    'profile', default=True,
    help='Enable programmatic profile with xprof.')

flags.DEFINE_bool(
    'profile_first_eval', default=True,
    help='Enable programmatic profile with xprof for eval.')

flags.DEFINE_integer(
    'repeat_experiment', default=1, help=('Number of runs'))

flags.DEFINE_integer(
    'profile_duration', default=15, help=('Xprof profile duration'))

flags.DEFINE_integer(
    'profile_latency', default=15, help=('When to start profiling.'))

flags.DEFINE_integer(
    'num_partitions', default=1, help=('Number of partitions in SPMD.'))

flags.DEFINE_integer(
    'num_eval_partitions', default=1, help=('Number of partitions in SPMD.'))
flags.DEFINE_string(
    'experiment_name', help='name of the experiment', default='')
flags.DEFINE_string(
    'experiment_dir', help='directory of the experiment', default='')

# Adds jax_log_compiles flag to print compilation logs on the jax side.
config.parse_flags_with_absl()
FLAGS = flags.FLAGS


mypmap = partial(jax.pmap, axis_name='hosts')


@mypmap
def host_psum(x):
  return jax.lax.psum(x, 'hosts')


def per_host_sum_pmap(in_tree):
  """Execute sum on in_tree's leaves over ICI."""
  ldc = jax.local_device_count()
  def pre_pmap(x):
    y = onp.zeros((ldc, *x.shape), dtype=x.dtype)
    y[0] = x
    return y
  def post_pmap(x):
    return jax.device_get(x)[0]
  return jax.tree_map(post_pmap, host_psum(jax.tree_map(pre_pmap, in_tree)))


def construct_run_config():
  """Construct the run config parameters.

  Returns:
    A dictionary containing run parameters.
  """

  if FLAGS.use_eval_device_loop:
    # Eval device loop does not support spmd.
    assert FLAGS.num_eval_partitions == 1
  num_cores = jax.local_device_count() * jax.host_count()
  num_replicas = num_cores // FLAGS.num_partitions
  num_eval_replicas = num_cores // FLAGS.num_eval_partitions
  dtype = jnp.bfloat16 if FLAGS.use_bfloat16 else jnp.float32

  # steps_per_epoch = ceil(168 / 32) = 6 for bs=32
  num_steps_per_epoch = math.ceil(FLAGS.num_train_images / FLAGS.batch_size)
  # 192 for 6 * 32 for bs=32
  samples_per_epoch = num_steps_per_epoch * FLAGS.batch_size
  # Stil provide the parameters as original,
  # max 10K epochs, 10K * 168 samples to converge,
  # first epoch is at 1000, and evaluate every 20 epochs.
  # Warmup epochs is 1000, meaning 168 & 1000 samples.
  # start_eval_at = 1000
  # epochs = 10000
  # evaluate_every = 20
  macro_step_sizes = []
  # first_eval_epoch = 875, ceil(168 * 1000 / 198)
  first_eval_epoch = math.ceil(FLAGS.num_train_images * FLAGS.start_eval_at /
                               samples_per_epoch)
  # first_eval_step = 875 * 6
  first_eval_step = first_eval_epoch * num_steps_per_epoch

  # later_eval_epoch_frequency = 18, ceil(168 * 20 / 192)
  later_eval_epoch_frequency = math.ceil(FLAGS.num_train_images *
                                         FLAGS.evaluate_every /
                                         samples_per_epoch)

  # later_eval_step_frequency = 18 * 6 = 108
  later_eval_step_frequency = later_eval_epoch_frequency * num_steps_per_epoch
  # macro_step_sizes = [5250, 108]
  macro_step_sizes = [first_eval_step, later_eval_step_frequency]

  # 6 steps are called an epoch

  # No crosshost spmd for eval.
  host_eval_batch_size = FLAGS.eval_batch_size // jax.host_count()
  assert host_eval_batch_size > 0
  replica_eval_batch_size = FLAGS.eval_batch_size // num_eval_replicas
  assert replica_eval_batch_size > 0
  num_host_eval_replicas = jax.local_device_count() // FLAGS.num_eval_partitions
  assert num_host_eval_replicas > 0
  local_num_replicas = jax.local_device_count() // FLAGS.num_partitions
  local_num_replicas = max(1, local_num_replicas)
  hosts_per_replicas = FLAGS.num_partitions // jax.local_device_count()
  hosts_per_replicas = max(1, hosts_per_replicas)
  replica_batch_size = FLAGS.batch_size // num_replicas
  replicas_per_hosts = jax.local_device_count() // FLAGS.num_partitions
  replicas_per_hosts = max(1, replicas_per_hosts)
  host_batch_size = replicas_per_hosts * replica_batch_size

  num_eval_steps = math.ceil(
      input_reader.NUM_SLIDING_WINDOWS / FLAGS.eval_batch_size)
  return dict(
      use_train_device_loop=FLAGS.use_train_device_loop,
      use_eval_device_loop=FLAGS.use_eval_device_loop,
      make_sliding_windows_in_dataset=True,
      num_eval_images=FLAGS.num_eval_images,
      eval_score_fn_bs=FLAGS.eval_score_fn_bs,
      num_eval_steps=num_eval_steps,
      # Global batch size for eval, has to be multiple of host_count.
      eval_batch_size=FLAGS.eval_batch_size,
      # Per host eval batch size.
      host_eval_batch_size=host_eval_batch_size,
      # Per replica eval batch size.
      replica_eval_batch_size=replica_eval_batch_size,
      # Number of global eval replicas
      num_eval_replicas=num_eval_replicas,
      num_host_eval_replicas=num_host_eval_replicas,
      num_train_images=FLAGS.num_train_images,
      num_eval_partitions=FLAGS.num_eval_partitions,

      num_partitions=FLAGS.num_partitions,
      use_spatial_partitioning=FLAGS.num_partitions > 1,
      local_num_replicas=local_num_replicas,
      hosts_per_replicas=hosts_per_replicas,
      num_cores=num_cores,
      macro_step_sizes=macro_step_sizes,
      num_steps_per_epoch=num_steps_per_epoch,
      samples_per_epoch=samples_per_epoch,
      num_local_devices=jax.local_device_count(),
      device_batch_size=replica_batch_size,
      host_batch_size=host_batch_size,
      num_replicas=num_replicas,
      data_dir=FLAGS.data_dir,
      epochs=FLAGS.epochs,
      batch_size=FLAGS.batch_size,
      layout=FLAGS.layout,
      input_shape=FLAGS.input_shape,
      input_shape_without_channel=FLAGS.input_shape[:-1],
      val_input_shape=FLAGS.val_input_shape,
      val_input_shape_without_channel=FLAGS.val_input_shape[:-1],
      seed=FLAGS.seed,
      exec_mode=FLAGS.exec_mode,
      use_bfloat16=FLAGS.use_bfloat16,
      optimizer=FLAGS.optimizer,
      learning_rate=FLAGS.learning_rate,
      init_learning_rate=FLAGS.init_learning_rate,
      lr_warmup_epochs=FLAGS.lr_warmup_epochs,
      lr_decay_epochs=FLAGS.lr_decay_epochs,
      lr_decay_factor=FLAGS.lr_decay_factor,
      lamb_beta1=FLAGS.lamb_betas[0],
      lamb_beta2=FLAGS.lamb_betas[1],
      momentum=FLAGS.momentum,
      weight_decay=FLAGS.weight_decay,
      evaluate_every=FLAGS.evaluate_every,
      normalization=FLAGS.normalization,
      activation=FLAGS.activation,
      pad_mode=FLAGS.pad_mode,
      oversampling=FLAGS.oversampling,
      include_background=FLAGS.include_background,
      dtype=dtype,
      in_channels=1,
      n_class=3,
      shuffle_buffer_size=FLAGS.num_train_images,
      interleave_cycle_length=32,
      num_hosts=jax.host_count(),
      host_index=jax.host_id(),
      overlap=FLAGS.overlap,
      eval_mode='gaussian',
      padding_val=-2.2,
      eval_padding_mode='constant',  # to be used in eval sliding windows.
      use_fake_data=FLAGS.use_fake_data,
      fake_nan_data=False,
      use_fake_train_data=False,
      num_eval_passes=FLAGS.num_eval_passes,
      eval_image_indices=FLAGS.eval_image_indices,
      )


def main(argv):
  # BEGIN GOOGLE-INTERNAL
  xm.setup_work_unit()
  # END GOOGLE-INTERNAL
  del argv

  tf.enable_v2_behavior()
  params = construct_run_config()
  logging.info('Experiment params: %s', params)
  for _ in range(FLAGS.repeat_experiment):
    run_unet(params)


def run_unet(params):
  """Runs a single end to end unet experiment."""
  logging.info('params:%s', params)
  host_id = params['host_index']

  params['training_num_hosts'] = params['num_replicas']

  params['training_host_index'] = 2
  if FLAGS.seed >= 0:
    seed = FLAGS.seed
  else:
    seed = onp.uint32(time.time() if host_id == 0 else 0)
    seed = onp.int64(per_host_sum_pmap(seed))
  tf.random.set_seed(seed)

  train_dataloader, _ = data_loader.get_data_loaders(
      FLAGS.data_dir, params)
  train_dataset = train_dataloader(params)

  train_iterator = iter(train_dataset)
  for step in range(params['epochs']):
    my_ti = next(train_iterator)
    # pylint: disable=cell-var-from-loop
    my_ti = jax.tree_map(lambda x: x.numpy(), my_ti)
    # pylint: enable=cell-var-from-loop
    ti = per_host_sum_pmap(my_ti)
    ti = jax.tree_map(lambda x: x / params['num_hosts'], ti)
    for key in ['image', 'label']:
      my_ti[key] = my_ti[key] - ti[key]

      diff = math.fabs(onp.sum(my_ti[key]))
      if diff < 0.0001:
        logging.info('step:%s host:%s key:%s np.sum(my_ti[key]):%s',
                     step, host_id, key, diff)
      else:
        logging.error('step:%s host:%s key:%s np.sum(my_ti[key]):%s', step,
                      host_id, key, diff)


if __name__ == '__main__':
  app.run(main)

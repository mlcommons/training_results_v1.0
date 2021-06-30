# Lint as: python3
"""Unet3d main training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from concurrent.futures import thread
import math
import os
import socket
import time

from absl import app
from absl import flags
from absl import logging

import jax
from jax import config
from jax import random
import jax.numpy as jnp
from jax.util import partial

import numpy as onp
import tensorflow.compat.v2 as tf

import REDACTED.learning.deepmind.REDACTED.client.google as xm
from REDACTED import xprof_session
from REDACTED.mlperf.submissions.training.v1_0.models.mlp_log import mlp_log
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.data_loading import data_loader
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.models import losses
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.runtime import arguments  # pylint: disable=unused-import
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.runtime import inference
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.runtime import training


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

flags.DEFINE_integer(
    'num_eval_steps', default=1, help=('Number of eval steps.'))


# Adds jax_log_compiles flag to print compilation logs on the jax side.
config.parse_flags_with_absl()
FLAGS = flags.FLAGS

RUN_START = False
RUN_STOP = False
PROFILE_URL = None
TOTAL_STEPS = -1

QUALITY_THRESHOLD = 0.908
FINAL_SCORE = None
CONVERGENCE_EPOCHS = None

mypmap = partial(jax.pmap, axis_name='hosts')

printed_weights = {}


def print_funct(tensor_name):
  # FLAX nn.modules apply functions are called twice. Only print for the first
  # time, when the weights are initialized.
  if tensor_name in printed_weights:
    pass
  else:
    printed_weights[tensor_name] = 1
    mlp_log.mlperf_print(
        'weights_initialization', None, stack_offset=1,
        metadata={'tensor': tensor_name})


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


def xprof_start(device='REDACTED'):
  xprof = xprof_session.XprofSession()
  xprof.start_session(device_name=device,
                      enable_python_tracer=True,
                      host_trace_level=2)
  return xprof


def xprof_stop(xprof):
  global PROFILE_URL
  xprof_url = xprof.end_session_and_get_url(tag='')
  logging.info('Xprof profile is at %s', xprof_url)
  PROFILE_URL = xprof_url


def xprof_profile(start_after_sec=30, profile_time_sec=1,
                  device='REDACTED'):
  """Profiles single host with after start_after_sec for profile_time_sec.

  Args:
    start_after_sec: when to start profiling in sec.
    profile_time_sec: how long to profile in sec.
    device: string, one of ['', 'cpu', 'gpu', 'REDACTED', 'REDACTED'].
  """
  if device not in  ['', 'cpu', 'gpu', 'REDACTED', 'REDACTED']:
    logging.error('Incorrect device for profiling %s', device)
    return
  time.sleep(start_after_sec)
  xprof = xprof_start(device=device)
  time.sleep(profile_time_sec)
  xprof_stop(xprof)


def profile_with_xprof_on_background(profiler_thread, start_after_sec=30,
                                     profile_time_sec=1, device='REDACTED'):
  profiler_thread.submit(partial(xprof_profile, start_after_sec,
                                 profile_time_sec, device))


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

  return dict(
      use_train_device_loop=FLAGS.use_train_device_loop,
      use_eval_device_loop=FLAGS.use_eval_device_loop,
      make_sliding_windows_in_dataset=True,
      num_eval_images=FLAGS.num_eval_images,
      eval_score_fn_bs=FLAGS.eval_score_fn_bs,
      num_eval_steps=FLAGS.num_eval_steps,
      num_eval_batches=FLAGS.num_eval_steps * FLAGS.eval_batch_size,
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


def _sync_devices(x):
  return jax.lax.psum(x, 'i')


def sync_devices():
  """Creates a barrier across all hosts/devices."""
  jax.pmap(_sync_devices, 'i')(
      onp.ones(jax.local_device_count())).block_until_ready()


def write_experiment_summary(convergence_epochs, final_score, all_host_names):
  """Writes the experiment result."""
  outfile = os.path.join(FLAGS.experiment_dir, FLAGS.experiment_name)
  tf.io.gfile.makedirs(FLAGS.experiment_dir)
  out = tf.io.gfile.GFile(outfile, mode='w')
  global RUN_START, RUN_STOP, PROFILE_URL
  overal_time = None
  if RUN_START and RUN_STOP:
    overal_time = RUN_STOP - RUN_START
  host_csv = ''
  for host_name in all_host_names:
    host_csv += host_name + ' '
  content = '%s %s %s %s %s %s %s %s\n' % (
      FLAGS.experiment_name, convergence_epochs, final_score, RUN_START,
      RUN_STOP, overal_time, PROFILE_URL, host_csv)
  out.write(content)
  out.close()


def get_host_names():
  """Returns all host names for logging purposes."""
  my_host_name = socket.gethostname()
  num_hosts = jax.host_count()
  my_host_id = jax.host_id()
  max_host_char_length = 128
  all_host_names = onp.zeros([num_hosts, max_host_char_length], dtype=onp.int32)
  i = 0
  for c in my_host_name:
    all_host_names[my_host_id, i] = ord(c)
    i += 1
  all_host_names = per_host_sum_pmap(all_host_names)
  host_list = []
  for i in range(num_hosts):
    host_name = ''
    for c in all_host_names[i]:
      if c == 0:
        break
      host_name += chr(c)
    host_list.append(host_name.split('.')[0])
  return host_list


def main(argv):
  # BEGIN GOOGLE-INTERNAL
  xm.setup_work_unit()
  # END GOOGLE-INTERNAL
  del argv

  all_host_names = get_host_names()
  tf.enable_v2_behavior()
  params = construct_run_config()
  logging.info('Experiment params: %s', params)
  for _ in range(FLAGS.repeat_experiment):
    train_metrics, eval_metrics, final_score, convergence_epochs = run_unet(
        params)
    if FLAGS.experiment_dir and FLAGS.experiment_name:
      if jax.host_id() == 0:
        write_experiment_summary(convergence_epochs, final_score,
                                 all_host_names)
    del train_metrics, eval_metrics, final_score, convergence_epochs


def get_unet_model(rng, params, replicate_fn=None):
  """Returns mlperf model, state, loss and learning rate schedule fns."""
  jax_unet, state = training.get_unet_model(rng, params, print_funct)
  optimizer = training.create_optimizer(jax_unet, params,
                                        replicate=False)
  loss_fn, _ = losses.get_loss_fn(rng, params)
  learning_rate_schedule = training.create_learning_rate_scheduler(params)
  del jax_unet
  if replicate_fn:
    state = replicate_fn(state)
    optimizer = replicate_fn(optimizer)
    # Below ones does not have any parameter, so they should not have any
    # state.
    loss_fn = replicate_fn(loss_fn)
  return optimizer, state, loss_fn, learning_rate_schedule


def print_initial_mlperf_config(params, seed):
  """Prints MLPerf config."""
  mlp_log.mlperf_print('cache_clear', value=True)
  mlp_log.mlperf_print('init_start', value=None)
  mlp_log.mlperf_print('global_batch_size', params['batch_size'])
  mlp_log.mlperf_print('opt_name', value=FLAGS.optimizer)
  mlp_log.mlperf_print('opt_base_learning_rate', params['learning_rate'])
  mlp_log.mlperf_print('opt_learning_rate_warmup_epochs',
                       params['lr_warmup_epochs'])
  mlp_log.mlperf_print('opt_learning_rate_decay_boundary_epochs',
                       params['lr_decay_epochs'])

  mlp_log.mlperf_print('opt_learning_rate_decay_factor',
                       params['lr_decay_factor'])
  mlp_log.mlperf_print('opt_weight_decay', params['weight_decay'])
  mlp_log.mlperf_print('train_samples', FLAGS.num_train_images)
  mlp_log.mlperf_print('eval_samples', FLAGS.num_eval_images)
  mlp_log.mlperf_print('seed', int(seed))

  mlp_log.mlperf_print('opt_momentum', params['momentum'])
  mlp_log.mlperf_print('oversampling', params['oversampling'])
  mlp_log.mlperf_print('training_input_shape', params['input_shape'])
  mlp_log.mlperf_print('validation_input_shape', params['val_input_shape'])
  mlp_log.mlperf_print('validation_overlap', params['overlap'])

  mlp_log.mlperf_print('opt_learning_rate_warmup_factor', 1)

  mlp_log.mlperf_print('opt_initial_learning_rate',
                       params['init_learning_rate'])
  mlp_log.mlperf_print('submission_benchmark', 'unet3d')
  mlp_log.mlperf_print('gradient_accumulation_steps', 1)
  mlp_log.mlperf_print('samples_per_epoch', params['samples_per_epoch'])


def should_stop(mean_dice):
  return mean_dice >= FLAGS.quality_threshold


def _write_metrics(eval_metrics, train_metrics, host_epoch,
                   params, host_id, last_block_size,
                   next_block_size):
  """Logs the accuracy metrics."""

  try:
    del host_id
    global RUN_STOP, FINAL_SCORE, CONVERGENCE_EPOCHS
    if RUN_STOP and FLAGS.exec_mode == 'train':
      return
    train_metrics = jax.tree_map(jax.device_get, train_metrics)
    eval_metrics = jax.tree_map(jax.device_get, eval_metrics)

    if FLAGS.use_eval_device_loop:
      # Currently each replica computes the same score.
      eval_metrics['eval_score'] = jax.tree_map(lambda x: x[0],
                                                eval_metrics['eval_score'])
    # Sum list of [dice_score_0, dice_score1] pairs.
    eval_metrics['eval_score'] = jax.tree_map(onp.sum,
                                              eval_metrics['eval_score'])
    mean_dice = onp.sum(eval_metrics['eval_score']) / FLAGS.num_eval_images / 2

    FINAL_SCORE = mean_dice

    # The epochs are printed here are end epochs.
    # E.g, 1000, 1020, 1040, 1060.
    mlp_log.mlperf_print(
        'eval_accuracy', mean_dice, metadata={'epoch_num': host_epoch})
    # Print block stop for all evaluations, including when we reach target
    # accruact.
    # This will print block_stops for
    # first_epoch_num=1, epoch_count=1000,
    # first_epoch_num=1001, epoch_count=20,
    # first_epoch_num=1021, epoch_count=20, and so on.
    mlp_log.mlperf_print(
        'block_stop',
        None,
        metadata={'first_epoch_num': host_epoch - last_block_size + 1,
                  'epoch_count': last_block_size})
    if should_stop(mean_dice):
      mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
      RUN_STOP = time.time()
      CONVERGENCE_EPOCHS = host_epoch
    else:
      # This will print block_stops for
      # first_epoch_num=1001, epoch_count=20,
      # first_epoch_num=1021, epoch_count=20, and so on.
      mlp_log.mlperf_print(
          'block_start',
          None,
          metadata={
              'first_epoch_num': host_epoch + 1,
              'epoch_count': next_block_size
          })
    total_training_samples = params['epochs'] * FLAGS.num_train_images
    processed_samples = params['samples_per_epoch'] * host_epoch
    logging.info(
        '(Samples %s / %s), mean_dice:%s eval_metrics:%s',
        processed_samples, total_training_samples, mean_dice, eval_metrics)
  except Exception as thread_failure:  # pylint: disable=broad-except
    # The errors seems to be suppressed from the python threads.
    # If any error happens within python thread, print it here.
    RUN_STOP = time.time()
    logging.error('Problem while computing metrics:$%s', thread_failure)


def get_device_assignment(jax_devices, host_count, my_host_id, num_partitions):
  """Get device assignemnt permutation."""

  def bounds_from_last_device(device):
    x, y, z = device.coords
    return (x + 1), (y + 1), (z + 1), device.core_on_chip + 1

  single_host_dims = bounds_from_last_device(
      jax_devices[len(jax_devices) // host_count - 1])
  grid_dims = bounds_from_last_device(jax_devices[-1])
  topo_x, topo_y, topo_z, topo_c = grid_dims
  assert topo_x == 8
  assert topo_y == 8

  assert topo_c == 1
  assert num_partitions in (16, 32)
  assert single_host_dims[0] == 2
  assert single_host_dims[1] == 2

  coord_to_core = onp.ndarray(shape=(topo_x, topo_y, topo_z, topo_c),
                              dtype=int)
  coord_to_host = onp.ndarray(shape=(topo_x, topo_y, topo_z, topo_c),
                              dtype=int)

  for core_id, device in enumerate(jax_devices):
    host_id = device.host_id
    x, y, z = device.coords
    core_on_chip = device.core_on_chip
    coord_to_core[x][y][z][core_on_chip] = core_id
    coord_to_host[x][y][z][core_on_chip] = host_id
  devices = []
  local_devices = []
  for px in range(0, topo_x, 4):
    for py in range(0, topo_y, 2):
      for pz in range(0, topo_z, 2):
        walking_direction = [1, 1, 1]
        for z in range(2):
          for x in range(4):
            if walking_direction[1] < 0:
              x = 4 - x - 1
            for y in range(2):
              if walking_direction[2] < 0:
                y = 2 - y - 1
              ny = y + py
              nz = z + pz
              nx = x + px
              devices.append(jax_devices[coord_to_core[nx][ny][nz][0]])
              if coord_to_host[nx][ny][nz][0] == my_host_id:
                local_devices.append(jax_devices[coord_to_core[nx][ny][nz][0]])
            walking_direction[2] *= -1
          walking_direction[1] *= -1
        walking_direction[0] *= -1
  return devices, [local_devices]


def run_unet(params):
  """Runs a single end to end unet experiment."""
  logging.info('params:%s', params)
  global RUN_STOP, RUN_START, FINAL_SCORE, CONVERGENCE_EPOCHS, PROFILE_URL
  RUN_STOP = False
  RUN_START = False
  PROFILE_URL = None
  FINAL_SCORE = None
  CONVERGENCE_EPOCHS = None
  summary_thread = thread.ThreadPoolExecutor(1, 'summary')
  profiler_thread = thread.ThreadPoolExecutor(1, 'profiler')
  infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')
  host_id = params['host_index']
  # TODO: Improve device assignment.
  device_assignment = jax.devices()
  local_device_assignment = []
  for i in range(params['local_num_replicas']):
    local_device_assignment.append(
        jax.local_devices()[i * params['num_partitions']:(i + 1) *
                            params['num_partitions']])
  if FLAGS.space_filling_device_assignment:
    device_assignment, local_device_assignment = get_device_assignment(
        jax.devices(), params['num_hosts'], host_id, params['num_partitions'])

  for i in range(len(device_assignment)):
    d = device_assignment[i]
    logging.info('DeviceIdx:%d, coords:%s %s host:%d',
                 i, d.coords, d.core_on_chip, d.host_id)

  # Consecutive params['num_partitions'] cores work on the same replica.
  # We change the dataset host id below, so that they feed the same image.
  my_replica_id = host_id
  logging.info('I am host:%s', host_id)
  for r in range(params['num_replicas']):
    cores_in_replica = device_assignment[r * params['num_partitions']:(r + 1) *
                                         (params['num_partitions'])]
    hosts_in_replica = [d.host_id for d in cores_in_replica]
    logging.info('Replica:%s hosts_in_replica:%s', r, hosts_in_replica)
    if host_id in hosts_in_replica:
      my_replica_id = r
      # This host is the representative of the replica if it is the
      # first host in the replica.
      if host_id != hosts_in_replica[0]:
        # If the host is not the representative of the replica, then no need to
        # read input data from this host. Simply we generate fake data that
        # will be discarded. We make the fake data with all NaNs to make sure
        # that it will crash the model in the case of a mistake.
        params['use_fake_train_data'] = True
        params['fake_nan_data'] = True
        logging.info('Host:%s of replica r:%s will feed nan data', host_id, r)
      else:
        logging.info("Replica r:%s's input will be fed by host:%s", r, host_id)
  if params['num_replicas'] < params['num_hosts']:
    params['training_num_hosts'] = params['num_replicas']
    params['training_host_index'] = my_replica_id

  if FLAGS.seed >= 0:
    seed = FLAGS.seed
  else:
    seed = onp.uint32(time.time() if host_id == 0 else 0)
    seed = onp.int64(per_host_sum_pmap(seed))
  tf.random.set_seed(seed)
  rng = random.PRNGKey(seed)

  print_initial_mlperf_config(params, seed)

  optimizer, state, loss_fn, learning_rate_schedule = get_unet_model(
      rng, params)
  if params['exec_mode'] == 'train':
    if params['use_train_device_loop']:
      train_with_infeed_fn = training.get_train_with_infeed_fn(
          optimizer, state, loss_fn, learning_rate_schedule, params,
          device_assignment, local_device_assignment, infeed_pool)
    else:
      (ptrain_step, train_image_shape,
       train_label_shape) = training.make_parallel_train_step_fn_and_precompile(
           params, loss_fn, learning_rate_schedule, optimizer, state,
           device_assignment)
  sync_devices()
  unreplicated_optimizer = optimizer
  unreplicated_state = state
  evaluate_fn = inference.get_eval_fn(rng, unreplicated_optimizer.target,
                                      unreplicated_state, params, infeed_pool)

  train_dataloader, eval_dataloader = data_loader.get_data_loaders(
      FLAGS.data_dir, params)
  train_dataset = train_dataloader(params)
  eval_dataset = eval_dataloader(params)
  if FLAGS.profile and host_id == 0:
    profile_with_xprof_on_background(profiler_thread,
                                     FLAGS.profile_latency,
                                     FLAGS.profile_duration,
                                     device='REDACTED')
  if FLAGS.init_dummy_file:
    dummy_tensor = tf.io.read_file(FLAGS.init_dummy_file, name=None)
    del dummy_tensor

  sync_devices()
  RUN_START = time.time()
  macro_step_sizes = params['macro_step_sizes']
  num_steps_to_execute = macro_step_sizes[0]
  num_steps_per_epoch = params['num_steps_per_epoch']
  num_epochs_to_execute = num_steps_to_execute // num_steps_per_epoch
  last_block_size = num_epochs_to_execute

  mlp_log.mlperf_print('init_stop', None)
  mlp_log.mlperf_print('run_start', None)
  mlp_log.mlperf_print('block_start',
                       None,
                       metadata={'first_epoch_num': 1,
                                 'epoch_count': num_epochs_to_execute})

  train_iterator = iter(train_dataset)
  eval_iterator = iter(eval_dataset)
  epoch = 0
  train_metrics = {'train_loss': 0}
  eval_metrics = {}
  # While (epochs + num_epochs_to_execute) * 192 <= 10000 * 168
  while ((epoch + num_epochs_to_execute) * params['samples_per_epoch'] <=
         params['epochs'] * FLAGS.num_train_images):
    if params['exec_mode'] == 'train':
      logging.info('Running num_steps:%d training loops.', num_steps_to_execute)
      if params['use_train_device_loop']:
        optimizer, state, _ = train_with_infeed_fn(optimizer, state, epoch,
                                                   num_steps_to_execute,
                                                   train_iterator,
                                                   params)
      else:
        optimizer, state, host_loss = training.train_for_num_steps(
            ptrain_step, train_iterator, optimizer, state, num_steps_to_execute,
            epoch, train_image_shape, train_label_shape, params)
        train_metrics = {'train_loss': host_loss}

      epoch += num_epochs_to_execute
      last_block_size = num_epochs_to_execute
      num_steps_to_execute = macro_step_sizes[-1]
      num_epochs_to_execute = num_steps_to_execute // num_steps_per_epoch

    if FLAGS.profile and host_id == 0:
      FLAGS.profile = False
      profile_with_xprof_on_background(profiler_thread,
                                       FLAGS.profile_latency,
                                       FLAGS.profile_duration,
                                       device='REDACTED')
    unreplicated_optimizer = optimizer
    unreplicated_state = state
    overal_score = evaluate_fn(eval_iterator, unreplicated_optimizer.target,
                               unreplicated_state)
    # overal_score is at this point a list containing DeviceArrays of size 2.
    eval_metrics = {'eval_score': overal_score}
    write_epoch_metrics = partial(_write_metrics,
                                  eval_metrics, train_metrics, epoch,
                                  params, host_id, last_block_size,
                                  num_epochs_to_execute)
    summary_thread.submit(write_epoch_metrics)

    if params['exec_mode'] == 'evaluate':
      # Run the eval 5 times for better simulation and correctness checking.
      # e.g., all evals must return the same score.
      if epoch > 5:
        break
      epoch += 1
  summary_thread.shutdown()

  if not RUN_STOP:
    mlp_log.mlperf_print('run_stop', None, metadata={'status': 'abort'})
  profiler_thread.shutdown()
  infeed_pool.shutdown()
  del eval_dataloader
  return train_metrics, eval_metrics, FINAL_SCORE, CONVERGENCE_EPOCHS

if __name__ == '__main__':
  app.run(main)

"""Training loop for Unet3D model."""
import functools

from absl import flags
from absl import logging
from flax import jax_utils
from flax import nn
from flax import optim
import jax
from jax import lax
from jax import numpy as jnp
from jax.interpreters import sharded_jit
import numpy as np

from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.models import unet3d

flags.DEFINE_bool('reduce_gradients_in_bf16', True, 'Whether to use bfloat16 '
                  'for gradient all-reduce.')
FLAGS = flags.FLAGS


def get_unet_model(key, params, print_func=None):
  """Get unet3D model.

  Args:
    key: random.PRNGKey(seed)
    params: Config parameter dictionary.
    print_func: Function to call for reporting weight initializations.

  Returns:
    jax_model: unet3d model
    initial state
  """
  device_batch_size = params['device_batch_size']
  train_data_shape = params['input_shape']
  dtype = params['dtype']
  in_channels = params['in_channels']
  n_class = params['n_class']
  normalization = params['normalization']
  activation = params['activation']
  if print_func:
    unet_model_def = unet3d.Unet3D.partial(in_channels=in_channels,
                                           n_class=n_class,
                                           normalization=normalization,
                                           activation=activation,
                                           print_func=print_func)
  else:
    unet_model_def = unet3d.Unet3D.partial(in_channels=in_channels,
                                           n_class=n_class,
                                           normalization=normalization,
                                           activation=activation)

  input_shape = (device_batch_size, *train_data_shape)
  with nn.stateful() as init_state:
    _, model = unet_model_def.create_by_shape(key, [(input_shape, dtype)])
  return model, init_state


def create_optimizer(model, params, replicate=True):
  """Create optimizer used for training model.

  Args:
    model: JAX model to add optimizer to
    params: Config parameter dictionary.
    replicate: Whether to replicate the optimizer or not.

  Returns:
    optimizer: model with Adam/LAMB Optimizer to be used for training
  """
  learning_rate = params['learning_rate']
  weight_decay = params['weight_decay']
  # Optimizer parameters are set from the baseline here:
  # https://REDACTED.com/94TVkT4t5koprX8
  # http://REDACTED/_PwqK770Ju1

  if params['optimizer'] == 'adam':
    optimizer_def = optim.Adam(
        learning_rate=learning_rate,
        eps=1e-8,
        beta1=0.9,
        beta2=0.999,
        weight_decay=weight_decay)
  elif params['optimizer'] == 'sgd':
    momentum = params['momentum']
    optimizer_def = optim.Momentum(
        learning_rate=learning_rate,
        beta=momentum,
        weight_decay=weight_decay,
        nesterov=True)
  elif params['optimizer'] == 'lamb':
    lamb_beta1 = params['lamb_beta1']
    lamb_beta2 = params['lamb_beta2']
    optimizer_def = optim.LAMB(
        learning_rate=learning_rate,
        beta1=lamb_beta1,
        beta2=lamb_beta2,
        weight_decay=weight_decay,
        eps=1e-6)
  else:
    raise ValueError('Unexpected optimizer.')
  optimizer = optimizer_def.create(model)
  if replicate:
    optimizer = jax_utils.replicate(optimizer)
  return optimizer


def create_learning_rate_scheduler(params):
  """Create scheduler for learning rate."""
  # e.g., lr_warmup_epochs = 1000
  lr_warmup_epochs = params['lr_warmup_epochs']
  if lr_warmup_epochs <= 0:
    lr_warmup_epochs = -1
  # num_train_images=168
  # --> lr_warmup_samples=168000
  lr_warmup_samples = lr_warmup_epochs * params['num_train_images']
  # init_learning_rate = 0.0001
  init_learning_rate = params['init_learning_rate']
  # learning_rate = 4
  learning_rate = params['learning_rate']
  # lr_decay_factor = 0.9
  lr_decay_factor = params['lr_decay_factor']
  # lr_decay_epochs = [1500, 2000]
  lr_decay_epochs = params['lr_decay_epochs']
  # ---> lr_decay_samples = [252000, 336000]
  lr_decay_samples = lr_decay_epochs * params['num_train_images']
  # num_steps_per_epoch = 6, for bs=32
  num_steps_per_epoch = params['num_steps_per_epoch']
  lr_decay_samples = np.array(lr_decay_samples)

  def lr_schedule(step):
    """Step to learning rate function."""
    # If step = 4000
    # --> epoch= 666
    epoch = step // num_steps_per_epoch
    # --> processed_samples_in_epoch = 666 * 192 = 127872
    processed_samples = epoch * params['samples_per_epoch']
    # BS=32, samples_per_epoch=192, Epoch 10 -->
    # processed_samples_in_epoch = 1920
    # scale_factor will be 1920 / 16800
    # --> scale_factor 127872 / 168000 --> 0.761
    # (similar to the ratio of 666 / 875)
    warmup_lr = (init_learning_rate + (learning_rate - init_learning_rate) *
                 processed_samples / lr_warmup_samples)

    # If step = 9200
    # --> epoch = 1533 --> processed_samples == 1533 * 192 = 294336
    # np.sum(294336 < [252000, 336000] ) = 1
    num_milestones = jnp.sum((lr_decay_samples < processed_samples
                              ).astype(jnp.int32))
    # decay_lr = 0.9 ** 1
    decay_lr = learning_rate * (lr_decay_factor**num_milestones)
    return jax.lax.cond(processed_samples <= lr_warmup_samples,
                        warmup_lr, lambda x: x, decay_lr, lambda x: x)

  return lr_schedule


def make_parallel_train_step_fn_and_precompile(params,
                                               loss_fn,
                                               learning_rate_schedule,
                                               optimizer,
                                               state,
                                               device_assignment=None):
  """Returns jitted training step fn."""
  s_train_step = functools.partial(
      train_step,
      loss_fn=loss_fn,
      learning_rate_schedule=learning_rate_schedule,
      params=params)
  if params['num_partitions'] > 1:
    image_partition = sharded_jit.PartitionSpec(1, 1, params['num_partitions'],
                                                1, 1)

    label_partition = sharded_jit.PartitionSpec(1, 1, params['num_partitions'],
                                                1, 1)
    if params['hosts_per_replicas'] > 1:
      local_image_partition = sharded_jit.PartitionSpec(
          1, 1, jax.local_device_count(), 1, 1)

      local_label_partition = sharded_jit.PartitionSpec(
          1, 1, jax.local_device_count(), 1, 1)
      local_num_partitions = jax.local_device_count()
    else:
      local_image_partition = image_partition
      local_label_partition = label_partition
      local_num_partitions = params['num_partitions']
    train_partitions = ((None, None, image_partition, label_partition), None)
    local_train_partitions = ((None, None, local_image_partition,
                               local_label_partition), None)

    s_train_step = sharded_jit.sharded_jit(
        s_train_step,
        in_parts=train_partitions[0],
        out_parts=train_partitions[1],
        num_partitions=params['num_partitions'],
        local_in_parts=local_train_partitions[0],
        local_out_parts=local_train_partitions[1],
        local_num_partitions=local_num_partitions)

  train_image_shape = sharded_jit.PartitionSpec(
      params['device_batch_size'],
      params['input_shape'][0],
      params['input_shape'][1] * params['hosts_per_replicas'],
      params['input_shape'][2],
      params['input_shape'][3])
  train_label_shape = sharded_jit.PartitionSpec(
      params['device_batch_size'],
      params['input_shape'][0],
      params['input_shape'][1] * params['hosts_per_replicas'],
      params['input_shape'][2], params['input_shape'][3])
  if device_assignment:
    ptrain_step = jax.pmap(s_train_step, axis_name='batch',
                           axis_size=params['num_replicas'],
                           devices=device_assignment,
                           global_arg_shapes=(None, None, train_image_shape,
                                              train_label_shape),
                           in_axes=(None, None, 0, 0),
                           out_axes=(None, None, 0))
  else:
    ptrain_step = jax.pmap(s_train_step, axis_name='batch',
                           axis_size=params['num_replicas'],
                           global_arg_shapes=(None, None, train_image_shape,
                                              train_label_shape),
                           in_axes=(None, None, 0, 0),
                           out_axes=(None, None, 0))

  train_image_shape = (params['local_num_replicas'],
                       params['device_batch_size'],
                       params['input_shape'][0],
                       params['input_shape'][1],
                       params['input_shape'][2],
                       params['input_shape'][3])
  train_label_shape = (params['local_num_replicas'],
                       params['device_batch_size'],
                       params['input_shape'][0],
                       params['input_shape'][1],
                       params['input_shape'][2],
                       params['input_shape'][3])

  inp = jnp.array(np.random.randn(*train_image_shape)).astype(params['dtype'])
  label = jnp.array(np.random.randint(2, size=train_label_shape)).astype(
      np.int32)

  ptrain_step(optimizer, state, inp, label)
  return ptrain_step, train_image_shape, train_label_shape


def get_device_assignment(num_partitions):
  """Get device assignemnt permutation."""
  if num_partitions == 1:
    return jax.devices(), [[d] for d in jax.local_devices()]

  def bounds_from_last_device(device):
    x, y, _ = device.coords
    return (x + 1), (y + 1)

  jax_devices = jax.devices()
  host_count = jax.host_count()
  # TODO: get per host bounds from JAX backend.
  per_host_x, per_host_y = bounds_from_last_device(jax.local_devices(0)[-1])
  device_map = np.ndarray(shape=(host_count, per_host_x, per_host_y, 2),
                          dtype=int)

  for core_id, device in enumerate(jax_devices):
    host_id = device.host_id
    x, y, _ = device.coords
    core_on_chip = device.core_on_chip
    device_map[host_id][x % per_host_x][y % per_host_y][core_on_chip] = core_id

  replicas_per_host = jax.local_device_count() // num_partitions
  inner_y = min(num_partitions // 2, 2)
  inner_x = (num_partitions // 2) // inner_y
  # Set inner_ring within each replica.
  permute = list(range(num_partitions // 2)) + list(
      range(num_partitions - 1, num_partitions // 2 - 1, -1))

  device_assignment = []
  local_device_assignment = []
  for host_id in range(host_count):
    for replica_id in range(replicas_per_host):
      x_start = replica_id * inner_x
      cores_in_replica = []
      for y in range(inner_y):
        for x in range(x_start, x_start + inner_x):
          for core_on_chip in range(2):
            core_id = device_map[host_id][x][y][core_on_chip]
            cores_in_replica.append(core_id)
      # Set inner_ring within each replica for better performance.
      cores_in_replica = [cores_in_replica[i] for i in permute]
      replica_devices = [jax_devices[i] for i in cores_in_replica]
      if host_id == jax.host_id():
        local_device_assignment.append(replica_devices)
      device_assignment.extend(replica_devices)

  return device_assignment, local_device_assignment


def train_step(optimizer, state, features, labels, loss_fn,
               learning_rate_schedule, params):
  """Single training iteration.

  Args:
    optimizer: An instance of flax.optom.Optimizer.
    state: An instance of flax.nn.Collection.
    features: Input image batch for the iteration.
    labels: Input label batch for the iteration.
    loss_fn: Dice loss fn.
    learning_rate_schedule: Fn that returns a learning rate for a given step.
    params: A dictionary of run config parameters.

  Returns:
    An instance of the updated flax.optom.Optimizer and updated state.
  """
  assert features.dtype == params['dtype']
  logging.info('Feature Shape:%s', features)
  logging.info('Label Shape:%s', labels)
  # Currently, we do not have deterministic input pipeline for random ops.
  # Each host within replica works on the same image, but their random crop is
  # not deterministic. For now, we feed [128, 128, 128] images from each host,
  # and below we discard from all hosts but the first one, and reshard it.
  if 'hosts_per_replicas' in params and params['hosts_per_replicas'] > 1:
    features = features[:, :, :params['input_shape'][1], ...]
    labels = labels[:, :, :params['input_shape'][1], ...]
    label_partition = sharded_jit.PartitionSpec(1, 1, params['num_partitions'],
                                                1, 1)
    feature_partition = sharded_jit.PartitionSpec(1, 1,
                                                  params['num_partitions'], 1,
                                                  1)
    features = sharded_jit.with_sharding_constraint(features, feature_partition)
    labels = sharded_jit.with_sharding_constraint(labels, label_partition)
    logging.info('Feature Shape after reshard:%s', features)
    logging.info('Label Shape after reshard:%s', labels)

  @jax.jit
  def forward_pass(model):
    """Unet model loss function.

    Args:
      model: An instance of flax.optom.Optimizer.target.

    Returns:
      Loss and new state.
    """

    with nn.stateful(state) as new_state:
      output = model(features)
    output = output.astype(jnp.float32)
    loss = loss_fn(output, labels)
    return loss, new_state

  lr = learning_rate_schedule(optimizer.state.step)
  @jax.jit
  def compute_gradient():
    loss, new_state, grads = optimizer.compute_gradient(forward_pass)
    return loss, new_state, grads
  loss, new_state, grads = compute_gradient()
  if FLAGS.reduce_gradients_in_bf16:
    grads = jax.tree_map(lambda x: x.astype(jnp.bfloat16), grads)
  grads = jax.lax.pmean(grads, 'batch')
  if FLAGS.reduce_gradients_in_bf16:
    grads = jax.tree_map(lambda x: x.astype(jnp.float32), grads)

  @jax.jit
  def apply_gradient():
    new_optimizer = optimizer.apply_gradient(grads, learning_rate=lr)
    return new_optimizer
  new_optimizer = apply_gradient()
  return new_optimizer, new_state, loss


def train_for_num_steps(ptrain_step, train_iterator, optimizer, state,
                        num_steps, current_epoch,
                        train_image_shape, train_label_shape, params):
  """Implements a train epoch or macro step."""
  for step in range(num_steps):
    with jax.profiler.StepTraceAnnotation(
        'train', step_num=step + current_epoch * params['num_steps_per_epoch']):
      input_data = next(train_iterator)
      input_data = jax.tree_map(lambda x: x.numpy(), input_data)
      # Reshape it to num_local_devices, device_bs, image...
      input_data['image'] = np.reshape(input_data['image'],
                                       train_image_shape)
      input_data['label'] = np.reshape(input_data['label'],
                                       train_label_shape)

      optimizer, state, loss = ptrain_step(optimizer, state,
                                           input_data['image'],
                                           input_data['label'])
  return optimizer, state, loss


def assert_expected_shapes_and_dtypes(ds_out, expected_shapes_dtypes):
  """Asserts that the dataset output matches with the expected shapes and dtypes."""

  for data_name in expected_shapes_dtypes.keys():
    logging.info('%s shape: %s, %s', data_name, ds_out[data_name].shape,
                 expected_shapes_dtypes[data_name][0])
    logging.info('%s dtype: %s, %s', data_name, ds_out[data_name].dtype,
                 expected_shapes_dtypes[data_name][1])
    ds_shape = list(ds_out[data_name].shape)
    expected_shape = list(expected_shapes_dtypes[data_name][0])
    assert ds_shape == expected_shape
    real_dtype = jax.dtypes.canonicalize_dtype(ds_out[data_name].dtype)
    expected_dtype = jax.dtypes.canonicalize_dtype(
        expected_shapes_dtypes[data_name][1])
    assert real_dtype == expected_dtype


def get_per_replica_input_shapes_and_dtypes(params, from_device=False):
  """Returns the expected infeed shapes and dtypes.

  Args:
    params: Parameter dictionary.
    from_device: whether the shapes read from TPU devices or from hosts.
      Shapes might be different from host and device side when a replica spans
      multiple hosts.
  Returns:
    A dictionary holding image and label shape and dtype info.
  """
  device_batch_size = params['device_batch_size']
  input_shape = params['input_shape']
  input_shape_5d = np.insert(input_shape, 0, device_batch_size)
  image_dtype = params['dtype']
  label_dtype = jax.numpy.int32

  if params['hosts_per_replicas'] > 1 and from_device:
    # B, H, W, D, C --> partition along W.
    partition_axis = 2
    input_shape_5d[partition_axis] *= params['hosts_per_replicas']
    return {
        'image': (input_shape_5d, image_dtype),
        'label': (input_shape_5d, label_dtype)
    }
  else:
    return {
        'image': (input_shape_5d, image_dtype),
        'label': (input_shape_5d, label_dtype)
    }


def train_device_loop(optimizer, state, current_step, end_step, out_loss,
                      loss_fn, learning_rate_schedule, params):
  """Runs training for an epoch."""
  input_shapes_and_dtype = get_per_replica_input_shapes_and_dtypes(
      params, from_device=True)
  logging.info('Expected input_shapes_and_dtype:%s', input_shapes_and_dtype)
  def device_train_loop_cond(args):
    _, _, _, current_step, end_step = args
    return current_step < end_step

  def device_train_loop_body(args):
    optimizer, state, token, current_step, end_step = args
    if params['num_partitions'] > 1:
      img_part = sharded_jit.PartitionSpec(1, 1, params['num_partitions'], 1, 1)
      lbl_part = sharded_jit.PartitionSpec(1, 1, params['num_partitions'], 1, 1)
      partitions = (img_part, lbl_part)
      (features, labels), token = lax.infeed(
          token,
          shape=(jax.ShapedArray(input_shapes_and_dtype['image'][0],
                                 input_shapes_and_dtype['image'][1]),
                 jax.ShapedArray(input_shapes_and_dtype['label'][0],
                                 input_shapes_and_dtype['label'][1])),
          partitions=partitions)
    else:
      (features, labels), token = lax.infeed(
          token,
          shape=(jax.ShapedArray(input_shapes_and_dtype['image'][0],
                                 input_shapes_and_dtype['image'][1]),
                 jax.ShapedArray(input_shapes_and_dtype['label'][0],
                                 input_shapes_and_dtype['label'][1])))
    # Make sure the input shapes and dtypes are as expected.
    assert_expected_shapes_and_dtypes({
        'image': features,
        'label': labels
    }, input_shapes_and_dtype)
    optimizer, state, _ = train_step(optimizer, state, features, labels,
                                     loss_fn, learning_rate_schedule, params)
    current_step += 1
    return optimizer, state, token, current_step, end_step

  token = lax.create_token(current_step)
  optimizer, state, _, current_step, _ = lax.while_loop(
      device_train_loop_cond, device_train_loop_body,
      (optimizer, state, token, current_step, end_step))
  # out_loss is added so that we have an in_axes=0 argument.
  # add a dummy operation here, since we have seen issues where a parameter
  # passed directly as output.
  out_loss = out_loss + current_step
  return optimizer, state, current_step, out_loss


def reshape_for_local_replicas(np_tensor, num_local_replicas):
  tensor_shape = np_tensor.shape
  new_tensor_shape = [num_local_replicas, -1]
  new_tensor_shape.extend(tensor_shape[1:])
  return np.reshape(np_tensor, new_tensor_shape)


def infeed_train_iter(infeed_pool, train_single_batch, infeed_devices, params,
                      check_input_shapes=False):
  """Infeeds all local devices with a single batch."""
  if params and check_input_shapes:
    # Make sure the input shapes and dtypes are as expected.
    assert len(infeed_devices) == train_single_batch['image'].shape[0]
    input_shapes_and_dtype = get_per_replica_input_shapes_and_dtypes(params)
    single_device_batch = jax.tree_map(lambda x: x[0], train_single_batch)
    assert_expected_shapes_and_dtypes(single_device_batch,
                                      input_shapes_and_dtype)
  num_local_partitions = min(params['num_partitions'], jax.local_device_count())
  for i in range(params['local_num_replicas']):
    replica_image = train_single_batch['image'][i]
    replica_label = train_single_batch['label'][i]
    partition_axis = 2
    axis_size = replica_image.shape[partition_axis]
    chunk_size = axis_size // num_local_partitions
    replica_image_shards = [
        replica_image[:, :, i:i + chunk_size]
        for i in range(0, axis_size, chunk_size)
    ]
    replica_label_shards = [
        replica_label[:, :, i:i + chunk_size]
        for i in range(0, axis_size, chunk_size)
    ]
    replica_devices = infeed_devices[i]
    for local_part in range(num_local_partitions):
      img_shard = replica_image_shards[local_part]
      lbl_shard = replica_label_shards[local_part]
      device = replica_devices[local_part]
      infeed_pool.submit(
          functools.partial(device.transfer_to_infeed, (img_shard, lbl_shard)))


def get_train_with_infeed_fn(optimizer, state, loss_fn, learning_rate_schedule,
                             params, device_assignment,
                             local_device_assignment,
                             infeed_pool):
  """Returns train fn with infeed."""
  train_epoch = functools.partial(
      train_device_loop,
      loss_fn=loss_fn,
      learning_rate_schedule=learning_rate_schedule,
      params=params)

  if params['num_partitions'] > 1:
    if params['hosts_per_replicas'] > 1:
      train_epoch = sharded_jit.sharded_jit(
          train_epoch,
          in_parts=(None, None, None, None, None),
          out_parts=(None, None, None, None),
          num_partitions=params['num_partitions'],
          local_in_parts=(None, None, None, None, None),
          local_out_parts=(None, None, None, None),
          local_num_partitions=jax.local_device_count())
    else:
      train_epoch = sharded_jit.sharded_jit(
          train_epoch,
          in_parts=(None, None, None, None, None),
          out_parts=(None, None, None, None),
          num_partitions=params['num_partitions'])

  p_train_epoch = jax.pmap(
      train_epoch, in_axes=(None, None, None, None, 0),
      out_axes=(None, None, None, 0),
      axis_size=params['num_replicas'],
      axis_name='batch',
      devices=device_assignment)
  num_local_replicas = params['local_num_replicas']
  # This should just compile and not execute.
  precompile_device_step = np.int32(0)
  end_step = np.int32(0)
  dummy_loss = np.zeros((num_local_replicas,), dtype=np.float32)

  p_train_epoch(optimizer, state, precompile_device_step, end_step, dummy_loss)
  infeed_devices = local_device_assignment
  device_reshape = functools.partial(
      reshape_for_local_replicas, num_local_replicas=num_local_replicas)

  def train_with_infeed(optimizer,
                        state,
                        epoch,
                        num_steps,
                        train_iterator,
                        params):
    dummy_loss = np.zeros((num_local_replicas,), dtype=np.float32)
    device_step = np.int32(0)
    end_step = np.int32(num_steps)
    optimizer, state, device_step, _ = p_train_epoch(optimizer, state,
                                                     device_step,
                                                     end_step, dummy_loss)
    for s in range(num_steps):
      with jax.profiler.StepTraceAnnotation(
          'train', step_num=s + epoch * params['num_steps_per_epoch']):
        train_ds_output = next(train_iterator)
        train_ds_output = jax.tree_map(lambda x: x.numpy(), train_ds_output)
        train_single_batch = jax.tree_map(device_reshape, train_ds_output)
        infeed_train_iter(infeed_pool, train_single_batch, infeed_devices,
                          params)
    return optimizer, state, device_step
  return train_with_infeed

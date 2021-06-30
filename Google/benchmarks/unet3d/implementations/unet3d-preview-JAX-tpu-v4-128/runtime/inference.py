"""Jax Unet inference implementation."""

from concurrent.futures import thread
import functools
import math

from absl import logging
from flax import nn
import jax
from jax.interpreters import sharded_jit
import jax.numpy as jnp
import numpy as np

from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.data_loading import input_reader
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.helpers import helpers
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.models import losses


def multi_device_score_fn(device_results, labels, is_real_input):
  """Returns the sum of the scores."""
  device_results = device_results.astype(jnp.bfloat16)
  labels = labels.astype(jnp.int8)
  device_results = jax.lax.psum(device_results, axis_name="cores")
  labels = jax.lax.pmax(labels, axis_name="cores")
  device_scores = losses.dice_score(device_results, labels)
  return jnp.sum(device_scores * is_real_input.astype(jnp.float32), axis=0)


def precompile_score_fn(params):
  """First sums up devices results from different cores, masks it and calls original score fn."""
  del params
  num_partitions = jax.local_device_count()
  num_total_devices = jax.device_count()
  result_partition = sharded_jit.PartitionSpec(1, num_partitions, 1, 1, 1)
  label_partition = sharded_jit.PartitionSpec(1, num_partitions, 1, 1, 1)

  s_score_fn = sharded_jit.sharded_jit(
      multi_device_score_fn,
      num_partitions=num_partitions,
      in_parts=(result_partition, label_partition, None),
      out_parts=None)
  p_score_fn = jax.pmap(
      s_score_fn,
      axis_name="cores",  # 1 replica per host.
      axis_size=num_total_devices // num_partitions,
      in_axes=(0, None, None),
      donate_argnums=(0, 1, 2))
  return p_score_fn


def reshape_for_local_replicas(np_tensor, num_local_devices):
  tensor_shape = np_tensor.shape
  new_tensor_shape = [num_local_devices, -1]
  new_tensor_shape.extend(tensor_shape[1:])
  return np.reshape(np_tensor, new_tensor_shape)


def eval_core_fn(image, crop_masks, norm_maps, model, state, norm_patch,
                 params):
  """Applies forward pass of the model and updates."""
  assert image.dtype == params["dtype"]
  assert crop_masks.dtype == jnp.int8

  assert norm_maps.dtype == params["dtype"]
  assert norm_patch.dtype == params["dtype"]
  # TODO: We might add this as a constant for TPUs.
  # norm_patch = helpers.get_norm_patch(params)
  with nn.stateful(state, mutable=False):
    model_out = model(image)
    assert model_out.dtype == params["dtype"]
    model_out_scale = model_out * norm_patch * crop_masks / norm_maps
    assert model_out_scale.dtype == params["dtype"]
  return model_out_scale


def make_eval_core_fn(params):
  """Creates an core eval fn to be called from sliding windows."""
  eval_core_function = functools.partial(eval_core_fn,
                                         params=params)
  num_partitions = params["num_eval_partitions"]
  num_eval_replicas = params["num_eval_replicas"]
  if num_partitions > 1:
    image_partition = sharded_jit.PartitionSpec(1, num_partitions,
                                                1, 1, 1)
    in_partitions = (image_partition, None, None, None, None, None)
    s_eval_core = sharded_jit.sharded_jit(eval_core_function,
                                          in_parts=in_partitions,
                                          out_parts=None)
  else:
    s_eval_core = eval_core_function
  peval_core_fn = jax.pmap(s_eval_core,
                           axis_size=num_eval_replicas,
                           in_axes=(0, 0, 0, None, None, None))

  return peval_core_fn


def merge_result_on_cpu(global_result, global_labels,
                        roi_results, roi_labels,
                        image_indices, crop_locations, is_padding, params):
  """Merges the roi region result."""
  num_local_replicas = params["num_host_eval_replicas"]
  host_eval_batch_size = params["host_eval_batch_size"]
  roi_shape = params["val_input_shape_without_channel"]
  roi_results = jax.device_get(roi_results)
  for device_id in range(num_local_replicas):
    for batch_id in range(host_eval_batch_size):

      image_id = image_indices[device_id][batch_id]
      crop_loc = crop_locations[device_id][batch_id]
      if is_padding[device_id][batch_id]:
        continue
      i, j, k, _ = crop_loc
      global_result[image_id, i:(roi_shape[0] + i), j:(roi_shape[1] + j),
                    k:(roi_shape[2] + k), :] += roi_results[device_id][batch_id]

      # Overwrite the labels.
      global_labels[image_id, i:(roi_shape[0] + i), j:(roi_shape[1] + j),
                    k:(roi_shape[2] + k)] = roi_labels[device_id][batch_id]


def prepare_eval_for_window_batches(rng, params, optimizer, state):
  """Returns precompiled eval fns."""
  del rng
  eval_core_fn_jit = make_eval_core_fn(params)

  roi_shape = params["val_input_shape"]
  eval_replica_bs = params["replica_eval_batch_size"]
  num_host_eval_replicas = params["num_host_eval_replicas"]
  eval_image_shape = [
      num_host_eval_replicas, eval_replica_bs, roi_shape[0], roi_shape[1],
      roi_shape[2], roi_shape[3]
  ]
  image = np.zeros(eval_image_shape).astype(params["dtype"])
  crop_masks = np.zeros(eval_image_shape).astype(np.int8)
  norm_maps = np.zeros(eval_image_shape).astype(params["dtype"])
  norm_patch = helpers.get_norm_patch(params).astype(params["dtype"])

  eval_core_fn_jit(image, crop_masks, norm_maps, optimizer, state, norm_patch)
  pscore_fn = precompile_score_fn(params)
  score_fn_local_device_cnt = 1
  score_batch_size = params["eval_score_fn_bs"]
  eval_result_shape = [
      score_fn_local_device_cnt, score_batch_size,
      input_reader.MAX_EVAL_IMAGE_SHAPE[0],
      input_reader.MAX_EVAL_IMAGE_SHAPE[1],
      input_reader.MAX_EVAL_IMAGE_SHAPE[2], 3
  ]
  eval_result = np.zeros(eval_result_shape).astype(jax.numpy.bfloat16)
  label_shape = [score_batch_size,
                 input_reader.MAX_EVAL_IMAGE_SHAPE[0],
                 input_reader.MAX_EVAL_IMAGE_SHAPE[1],
                 input_reader.MAX_EVAL_IMAGE_SHAPE[2],
                 input_reader.MAX_EVAL_IMAGE_SHAPE[3]]
  label = np.zeros(label_shape).astype(np.int8)

  is_real_input = np.ones([score_batch_size, 1]).astype(np.float32)

  pscore_fn(eval_result, label, is_real_input)
  return eval_core_fn_jit, pscore_fn


def evaluate_host_loop(eval_iterator, model, state, peval_core_fn, score_fn,
                       params, async_merge=True):
  """Runs eval partially on CPU (due to memory reasons) for the epoch."""
  result_merger = thread.ThreadPoolExecutor(1, "result_merger")
  score_batch_size = params["eval_score_fn_bs"]
  num_eval_samples = params["num_eval_images"]
  padding_size = num_eval_samples % score_batch_size
  num_eval_samples_with_padding = padding_size + num_eval_samples
  is_real_input = np.concatenate(
      [np.ones([num_eval_samples, 1]),
       np.zeros([padding_size, 1])]).astype(np.float32)
  device_reshape = functools.partial(
      reshape_for_local_replicas,
      num_local_devices=params["num_host_eval_replicas"])
  norm_patch = helpers.get_norm_patch(params).astype(params["dtype"])
  num_eval_steps = params["num_eval_steps"]

  global_result = np.zeros(
      shape=(num_eval_samples_with_padding,
             *input_reader.MAX_EVAL_IMAGE_SHAPE[:-1], 3),
      dtype=jnp.bfloat16)

  global_labels = np.zeros(shape=(num_eval_samples_with_padding,
                                  *input_reader.MAX_EVAL_IMAGE_SHAPE),
                           dtype=np.int8)
  for step in range(num_eval_steps):
    with jax.profiler.StepTraceAnnotation("eval", step_num=step):
      eval_ds_output = next(eval_iterator)
      eval_ds_output = jax.tree_map(lambda x: x.numpy(), eval_ds_output)
      eval_ds_output = jax.tree_map(device_reshape, eval_ds_output)
      roi_result = peval_core_fn(eval_ds_output["image"],
                                 eval_ds_output["crop_masks"],
                                 eval_ds_output["norm_maps"],
                                 model, state, norm_patch)
      merge = functools.partial(merge_result_on_cpu,
                                global_result, global_labels,
                                roi_result, eval_ds_output["label"],
                                eval_ds_output["image_idx"],
                                eval_ds_output["crop_locations"],
                                eval_ds_output["is_padding"], params)
      if async_merge:
        result_merger.submit(merge)
      else:
        merge()
  result_merger.shutdown()
  scores = []

  for i in range(0, num_eval_samples_with_padding, score_batch_size):
    gr = np.expand_dims(global_result[i: i + score_batch_size], axis=0)
    # gnm = np.expand_dims(global_norm_map[i: i + score_batch_size], axis=0)
    # gl = np.expand_dims(global_labels[i: i + score_batch_size], axis=0)
    # gnm = global_norm_map[i: i + score_batch_size]
    gl = global_labels[i: i + score_batch_size]
    score_sums = score_fn(gr, gl, is_real_input[i:i + score_batch_size])
    scores.append(score_sums)
  return scores


def update_global_result(global_result, local_result, crop_loc, params):
  """Updates the global results with window result."""
  roi_shape = params["val_input_shape"]
  num_classes = global_result.shape[-1]

  i, j, k, _ = crop_loc
  result_portion = jax.lax.dynamic_slice(
      global_result,
      [i, j, k, 0],
      [roi_shape[0], roi_shape[1], roi_shape[2], num_classes])

  local_result_without_background = local_result[:, :, :, 1:]
  local_result_without_background -= local_result[:, :, :, :1]
  global_result = jax.lax.dynamic_update_slice(
      global_result,
      result_portion + local_result_without_background,
      [i, j, k, 0])
  return global_result


def update_label(global_label, local_label, crop_loc):
  """Updates the global label with window label."""
  i, j, k, _ = crop_loc
  global_label = jax.lax.dynamic_update_slice(global_label, local_label,
                                              [i, j, k, 0])
  return global_label


def get_per_replica_eval_shapes_and_dtypes(params):
  """Returns the expected shape and dtypes for eval step inputs."""
  eval_input = {}
  roi_shape = params["val_input_shape"]
  replica_eval_batch_size = params["replica_eval_batch_size"]

  shape_5d = [
      replica_eval_batch_size, roi_shape[0], roi_shape[1], roi_shape[2],
      roi_shape[3]
  ]

  image_dtype = jnp.bfloat16 if params["use_bfloat16"] else jnp.float32
  eval_input["image"] = (shape_5d, image_dtype)
  eval_input["label"] = (shape_5d, jnp.int8)
  eval_input["crop_masks"] = (shape_5d, jnp.int8)
  eval_input["norm_maps"] = (shape_5d, image_dtype)

  eval_input["image_idx"] = ([replica_eval_batch_size], jnp.int32)
  eval_input["crop_locations"] = ([replica_eval_batch_size, 4], jnp.int32)
  eval_input["is_padding"] = ([replica_eval_batch_size], jnp.bool_)
  return eval_input


def assert_expected_shape_and_dtypes(ds_out, expected_shape_dtypes):
  """Asserts that the dataset output matches with the expected eval shapes and dtypes."""

  for data_name in expected_shape_dtypes.keys():
    logging.info("Eval shape :%s :%s :%s", data_name, ds_out[data_name].shape,
                 expected_shape_dtypes[data_name][0])
    logging.info("Eval dtype :%s :%s :%s", data_name, ds_out[data_name].dtype,
                 expected_shape_dtypes[data_name][1])
    ds_shape = list(ds_out[data_name].shape)
    expected_shape = list(expected_shape_dtypes[data_name][0])
    assert ds_shape == expected_shape
    real_dtype = jax.dtypes.canonicalize_dtype(ds_out[data_name].dtype)
    expected_dtype = jax.dtypes.canonicalize_dtype(
        expected_shape_dtypes[data_name][1])
    assert real_dtype == expected_dtype


def infeed_eval_iter(infeed_pool, eval_single_batch, infeed_devices,
                     params=None):
  """Infeeds all local devices with a single batch."""
  if params:
    # Make sure the input is as we expect.
    expected_input_config = get_per_replica_eval_shapes_and_dtypes(params)
    single_device_batch = jax.tree_map(lambda x: x[0], eval_single_batch)
    assert len(infeed_devices) == eval_single_batch["image"].shape[0]
    assert_expected_shape_and_dtypes(single_device_batch, expected_input_config)

  for index, device in enumerate(infeed_devices):
    infeed_pool.submit(
        functools.partial(device.transfer_to_infeed,
                          (eval_single_batch["image"][index],
                           eval_single_batch["label"][index],
                           eval_single_batch["crop_masks"][index],
                           eval_single_batch["norm_maps"][index],
                           eval_single_batch["image_idx"][index],
                           eval_single_batch["crop_locations"][index],
                           eval_single_batch["is_padding"][index])))


# We do not necessariliy need current_eval_step, but we will use it for
# precompile.
def evaluate_device_loop(model, state, current_eval_step, replica_id, params):
  """Runs eval for the epoch."""
  expected_input_config = get_per_replica_eval_shapes_and_dtypes(params)
  num_eval_samples = params["num_eval_images"]
  num_classes = 2
  global_results = {}
  global_labels = {}
  result_shape = (*input_reader.MAX_EVAL_IMAGE_SHAPE[:-1], num_classes)
  label_shape = (*input_reader.MAX_EVAL_IMAGE_SHAPE[:-1], 1)
  norm_patch = helpers.get_norm_patch(params).astype(params["dtype"])

  eval_core = functools.partial(eval_core_fn, params=params)
  eval_core = jax.jit(eval_core)
  update_global_result_jit = functools.partial(update_global_result,
                                               params=params)
  update_global_result_jit = jax.jit(update_global_result_jit)
  update_label_jit = jax.jit(update_label)
  for i in range(num_eval_samples):
    global_labels[i] = jnp.zeros(label_shape, dtype=jnp.int8)
    global_results[i] = jnp.zeros(result_shape, dtype=jnp.bfloat16)

  roi_shape = params["val_input_shape"]
  bs = params["replica_eval_batch_size"]

  def device_eval_loop_cond(args):
    model, state, results, labels, token, eval_step = args
    del model, state, results, labels, token
    return eval_step < params["num_eval_steps"]

  def device_eval_loop_body(args):
    model, state, results, labels, token, eval_step = args
    eval_step = eval_step + 1
    (image, label, crop_masks, norm_maps, image_idx, crop_locations,
     is_padding), token = jax.lax.infeed(
         token,
         shape=(jax.ShapedArray(expected_input_config["image"][0],
                                expected_input_config["image"][1]),
                jax.ShapedArray(expected_input_config["label"][0],
                                expected_input_config["label"][1]),
                jax.ShapedArray(expected_input_config["crop_masks"][0],
                                expected_input_config["crop_masks"][1]),
                jax.ShapedArray(expected_input_config["norm_maps"][0],
                                expected_input_config["norm_maps"][1]),
                jax.ShapedArray(expected_input_config["image_idx"][0],
                                expected_input_config["image_idx"][1]),
                jax.ShapedArray(expected_input_config["crop_locations"][0],
                                expected_input_config["crop_locations"][1]),
                jax.ShapedArray(expected_input_config["is_padding"][0],
                                expected_input_config["is_padding"][1])))
    eval_ds_output = {
        "image": image,
        "label": label,
        "crop_masks": crop_masks,
        "norm_maps": norm_maps,
        "image_idx": image_idx,
        "crop_locations": crop_locations,
        "is_padding": is_padding
    }

    # Make sure the input is as we expect.
    assert_expected_shape_and_dtypes(eval_ds_output, expected_input_config)

    batch_result = eval_core(eval_ds_output["image"],
                             eval_ds_output["crop_masks"],
                             eval_ds_output["norm_maps"], model, state,
                             norm_patch)
    batch_result = batch_result.astype(jnp.bfloat16)
    eval_ds_output["label"] = eval_ds_output["label"].astype(jnp.int8)
    assert batch_result.shape == (bs, roi_shape[0], roi_shape[1], roi_shape[2],
                                  3)
    for k in range(bs):
      local_label = eval_ds_output["label"][k]
      image_id = eval_ds_output["image_idx"][k]
      is_padding = eval_ds_output["is_padding"][k].astype(jnp.int32)
      # If padding none of the below branches should match.
      image_id = jnp.int32(image_id + is_padding * 1000)
      crop_loc = eval_ds_output["crop_locations"][k]
      single_window_result = batch_result[k]
      for i in range(params["image_index_begin"], params["image_index_end"]):
        # pylint: disable=cell-var-from-loop, g-long-lambda
        def update_while_loop_cond(args):
          (image_idx, static_image_idx, _, _, _) = args
          return image_idx == jnp.int32(static_image_idx)

        def update_result_while_loop_body(args):
          (image_idx, static_image_idx, image_result, single_window_r,
           window_loc) = args
          image_result = update_global_result_jit(image_result,
                                                  single_window_r,
                                                  window_loc)
          image_idx = -1
          return (image_idx, static_image_idx, image_result, single_window_r,
                  window_loc)

        def update_label_while_loop_body(args):
          (image_idx, static_image_idx, label_result, single_window_label,
           window_loc) = args
          label_result = update_label_jit(label_result, single_window_label,
                                          window_loc)
          image_idx = -1
          return (image_idx, static_image_idx, label_result,
                  single_window_label, window_loc)
        static_i = jnp.int32(i)
        loop_result = jax.lax.while_loop(
            update_while_loop_cond,
            update_result_while_loop_body,
            (image_id, static_i, results[i], single_window_result, crop_loc))
        results[i] = loop_result[2]

        loop_result = jax.lax.while_loop(
            update_while_loop_cond,
            update_label_while_loop_body,
            (image_id, static_i, labels[i], local_label, crop_loc))
        labels[i] = loop_result[2]

        # pylint: enable=cell-var-from-loop, g-long-lambda
    return model, state, results, labels, token, eval_step

  token = jax.lax.create_token(current_eval_step)
  (model, state, results, labels, token, _) = jax.lax.while_loop(
      device_eval_loop_cond, device_eval_loop_body,
      (model, state, global_results, global_labels, token, current_eval_step))

  for i in range(params["image_index_begin"], params["image_index_end"]):
    results[i] = jax.lax.psum(results[i], axis_name="cores")
    labels[i] = jax.lax.pmax(labels[i], axis_name="cores")
  scores = []
  score_fn = jax.jit(losses.dice_score_withoout_background)

  logging.info("num_eval_replicas:%s num_eval_images:%s",
               params["num_eval_replicas"], params["num_eval_images"])
  if params["num_eval_replicas"] < params["num_eval_images"]:
    # If we have less replicas than the number of images, simply all replicas
    # will compute the all scores.
    for i in range(params["image_index_begin"], params["image_index_end"]):
      results[i] = jnp.expand_dims(results[i], axis=0)
      labels[i] = jnp.expand_dims(labels[i], axis=0)
      scores.append(score_fn(results[i], labels[i]))
    return scores
  # This is a function that will never be executed, via jax.lax.cond dead
  # branch. We add this, to make sure that all psums are executed before
  # computing dice_scores. Otherwise, the compiler may try to overlap the
  # psum computation with score computation, which is done by one core at a
  # time.
  def force_execution_of_all_psums():
    r = []
    for i in range(params["image_index_begin"], params["image_index_end"]):
      result = results[i]
      label = labels[i]
      result = jnp.expand_dims(result, axis=0)
      label = jnp.expand_dims(label, axis=0)
      r.append(score_fn(result, label))
    return r
  dummy_fn = jax.jit(force_execution_of_all_psums)

  def score_loop_body(x):
    result, label = x
    result = jnp.expand_dims(result, axis=0)
    label = jnp.expand_dims(label, axis=0)
    return score_fn(result, label)
  score_loop_body = jax.jit(score_loop_body)

  def compute_scores():
    r = []
    for image_idx in range(params["image_index_begin"],
                           params["image_index_end"]):
      r.append(jax.lax.cond(
          replica_id == image_idx,
          lambda x: score_loop_body(x),  # pylint: disable=unnecessary-lambda
          lambda x: jnp.zeros([1, 2]).astype(jnp.float32),
          (results[image_idx], labels[image_idx])))
    return r
  compute_scores = jax.jit(compute_scores)
  args = (results, labels)
  scores = jax.lax.cond(
      replica_id == -1,  # replica id will never be -1
      lambda x: dummy_fn(),  # pylint: disable=unnecessary-lambda
      lambda x: compute_scores(),
      args)

  scores = [jax.lax.psum(r, axis_name="cores") for r in scores]
  return scores


def get_eval_fn(rng, optimizer, state, params, infeed_pool):
  """Returns evaluate fn."""
  if params["use_eval_device_loop"]:

    per_eval_image_size = int(
        math.ceil(params["num_eval_images"] / params["num_eval_passes"]))
    eval_fns = []
    replica_ids = np.arange(jax.local_device_count() * jax.host_id(),
                            jax.local_device_count() *
                            (jax.host_id() + 1)).astype(np.int32)

    for i in range(0, params["num_eval_images"], per_eval_image_size):
      params["image_index_begin"] = i
      params["image_index_end"] = min(i + per_eval_image_size,
                                      params["num_eval_images"])
      logging.info("Compiling eval for begin:%d, end:%d",
                   params["image_index_begin"], params["image_index_end"])
      eval_device_loop = functools.partial(evaluate_device_loop, params=params)
      p_eval_device_loop = jax.pmap(
          eval_device_loop, in_axes=(None, None, 0, 0), axis_name="cores")
      precompile_steps = (np.zeros([jax.local_device_count()], dtype=np.int64) +
                          np.int64(params["num_eval_steps"])).astype(np.int32)
      # This should just compile and not execute.
      p_eval_device_loop(optimizer, state, precompile_steps, replica_ids)
      eval_fns.append(p_eval_device_loop)

    infeed_devices = jax.local_devices()
    device_reshape = functools.partial(
        reshape_for_local_replicas,
        num_local_devices=params["num_host_eval_replicas"])
    def evaluate_with_infeed(eval_iterator, model, state):
      scores = []
      for i in range(params["num_eval_passes"]):
        current_step = np.zeros([jax.local_device_count()], dtype=np.int32)
        p_eval_device_loop = eval_fns[i]
        score = p_eval_device_loop(model, state, current_step, replica_ids)
        for s in range(params["num_eval_steps"]):
          with jax.profiler.StepTraceAnnotation("eval", step_num=s):
            eval_ds_output = next(eval_iterator)
            eval_ds_output = jax.tree_map(lambda x: x.numpy(), eval_ds_output)
            eval_single_batch = jax.tree_map(device_reshape, eval_ds_output)
            infeed_eval_iter(infeed_pool, eval_single_batch, infeed_devices)
        scores.extend(score)
      return scores
    return evaluate_with_infeed
  else:
    eval_core_fn_jit, pscore_fn = prepare_eval_for_window_batches(rng, params,
                                                                  optimizer,
                                                                  state)
    evaluate_fn = functools.partial(evaluate_host_loop,
                                    peval_core_fn=eval_core_fn_jit,
                                    score_fn=pscore_fn,
                                    params=params,
                                    async_merge=True)
  return evaluate_fn

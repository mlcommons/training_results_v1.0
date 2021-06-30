"""JAX implementation of losses in 3DUnet.

https://github.com/mmarcinkiewicz/training/blob/Add_unet3d/image_segmentation/unet3d/model/losses.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

import functools

from flax import nn
import jax

from jax import lax
import jax.numpy as jnp
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.models import losses_numpy


def cross_entropy_loss(logits: jnp.ndarray,
                       one_hot_labels: jnp.ndarray) -> jnp.ndarray:
  """Returns the cross entropy loss between some logits and some labels.

  Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.

  Returns:
    The cross entropy, averaged over the first dimension (samples).
  """
  log_softmax_logits = jax.nn.log_softmax(logits)
  loss = -jnp.sum(one_hot_labels * log_softmax_logits, axis=-1)
  return jnp.mean(loss)


class Dice(nn.Module):
  """Dice class."""

  def apply(self,
            prediction,
            target,
            to_onehot_y=True,
            to_onehot_x=False,
            use_softmax=True,
            use_argmax=False,
            include_background=False,
            layout="NDHWC"):
    smooth_nr = 1e-6
    smooth_dr = 1e-6

    if layout == "NCDHW":
      channel_axis = 1
      reduce_axis = tuple(list(range(2, len(prediction.shape))))
    else:
      channel_axis = -1
      reduce_axis = tuple(list(range(1, len(prediction.shape) - 1)))
    num_pred_ch = prediction.shape[channel_axis]

    if use_softmax:
      prediction = jax.nn.softmax(prediction, axis=channel_axis)
    elif use_argmax:
      prediction = jnp.argmax(prediction, axis=channel_axis)

    if to_onehot_y:
      target = to_one_hot(target, layout, channel_axis)

    if to_onehot_x:
      prediction = to_one_hot(prediction, layout, channel_axis)

    if not include_background:
      assert num_pred_ch > 1, \
          (f"To exclude background the prediction needs more than one channel. "
           f"Got {num_pred_ch}.")
      if layout == "NCDHW":
        target = target[:, 1:]
        prediction = prediction[:, 1:]
      else:
        target = target[..., 1:]
        prediction = prediction[..., 1:]

    assert (target.shape == prediction.shape), \
        (f"Target and prediction shape do not match. Target: ({target.shape}), "
         f"prediction: ({prediction.shape}).")

    intersection = jnp.sum(target * prediction, axis=reduce_axis)
    target_sum = jnp.sum(target, axis=reduce_axis)
    prediction_sum = jnp.sum(prediction, axis=reduce_axis)

    dice = (2.0 * intersection + smooth_nr) / (
        target_sum + prediction_sum + smooth_dr)
    return dice


def to_one_hot(array, layout, channel_axis):
  if len(array.shape) >= 5:
    array = jnp.squeeze(array, axis=channel_axis)
  array = jax.nn.one_hot(
      lax.convert_element_type(array, jnp.int32),
      num_classes=3,
      dtype=jnp.float32)
  if layout == "NCDHW":
    array = jnp.transpose(array, (0, 4, 1, 2, 3))
  return array


class DiceCELoss(nn.Module):
  """DiceCELoss class."""

  def apply(self, y_pred, y_true, to_onehot_y, use_softmax, layout,
            include_background=False):
    dice_fn = Dice.partial(to_onehot_y=to_onehot_y, use_softmax=use_softmax,
                           include_background=include_background)

    dice = 1.0 - jnp.mean(dice_fn(y_pred, y_true))
    if layout == "NCDHW":
      channel_axis = 1
    else:
      channel_axis = -1
    cross_entropy = cross_entropy_loss(y_pred,
                                       to_one_hot(y_true, layout, channel_axis))
    return (dice + cross_entropy) / 2


class DiceScore(nn.Module):
  """DiceScore class."""

  def apply(self,
            y_pred,
            y_true,
            to_onehot_y=True,
            use_argmax=True,
            layout="NDHWC",
            include_background=False,
            compute_mean_score=True):
    dice_fn = Dice.partial(
        to_onehot_y=to_onehot_y,
        to_onehot_x=True,
        use_softmax=False,
        use_argmax=use_argmax,
        layout=layout,
        include_background=include_background)
    dice_scores = dice_fn(y_pred, y_true)
    if compute_mean_score:
      return jnp.mean(dice_scores, axis=0)
    else:
      return dice_scores


def get_loss_fn(key, params, for_cpu=False, compute_mean_dice_score=True):
  """Get Unet DiceLoss and DiceScore fns.

  Args:
    key: random.PRNGKey(seed)
    params: Config parameter dictionary.
    for_cpu: If true, returns numpy implementation of the loss_fn and score_fn.
    compute_mean_dice_score: Whether to compute the mean scores or not for
      images.
  Returns:
    dice_loss_fn: dice loss fn.
    dice_score_fn: dice score fn.
  """
  layout = params["layout"]
  include_background = params["include_background"]
  dtype = params["dtype"]
  n_class = params["n_class"]

  device_batch_size = params["device_batch_size"]
  train_data_shape = params["input_shape"][:-1]
  input_shape = (device_batch_size, *train_data_shape, n_class)
  label_shape = (device_batch_size, *train_data_shape)

  if for_cpu:
    dice_loss_fn = functools.partial(
        losses_numpy.compute_dice_ce_loss,
        to_onehot_y=True,
        use_softmax=True,
        layout=layout,
        include_background=include_background)
    dice_score_fn = functools.partial(
        losses_numpy.compute_dice_score,
        to_onehot_y=True,
        use_argmax=True,
        layout=layout,
        include_background=include_background,
        compute_mean_score=compute_mean_dice_score)
  else:
    dice_loss_def = DiceCELoss.partial(
        to_onehot_y=True,
        use_softmax=True,
        layout=layout,
        include_background=include_background)
    _, dice_loss_fn = dice_loss_def.create_by_shape(key,
                                                    [(input_shape, dtype),
                                                     (label_shape, jnp.int32)])

    dice_score_def = DiceScore.partial(
        to_onehot_y=True,
        use_argmax=True,
        layout=layout,
        include_background=include_background,
        compute_mean_score=compute_mean_dice_score)
    _, dice_score_fn = dice_score_def.create_by_shape(key,
                                                      [(input_shape, dtype),
                                                       (label_shape, dtype)])
  return dice_loss_fn, dice_score_fn


def dice_score_withoout_background(prediction, target):
  """This is the special case of DiceScore where prediction has C - 1 channels.

  This is to reduce some memory usage from storing the global results. Each
  entry in the prediction now has C - 1 channels whee,
    new_prediction[..., :i] = old[..., :i + 1] - old[..., :0]

  So, when we check for argmax, we also check for max value, if less than 0,
  this means that it is background.

  Args:
    prediction: 5D numpy array. [Image, H, W, D, C - 1].
    target: 4D [Image, H, W, D], or 5D [Image, H, W, D, C, 1] target classes
      array.

  Returns:
    Dice score per image, [B]. Does not computes mean unlike others.
  """
  smooth_nr = 1e-6
  smooth_dr = 1e-6
  channel_axis = -1
  reduce_axis = tuple(list(range(1, len(prediction.shape) - 1)))
  # to reduce memory class scores are shrinked from 3 elements to 2 elements.
  # newscore[1] = oldscore[2] - old_score[0]
  # newscore[0] = oldscore[1] - old_score[0]
  # If max value is less than 0, this means that background has the max value.
  is_not_background = (jnp.max(prediction, axis=-1) > 0).astype(jnp.int8)

  # arg max will simply return 0, 1, 2. int8 is okay to use.
  prediction = jax.lax.argmax(prediction,
                              len(prediction.shape) + channel_axis,
                              jnp.int8)
  prediction = (prediction + 1) * is_not_background

  if len(prediction.shape) >= 5:
    prediction = jnp.squeeze(prediction, axis=channel_axis)

  # One hot will also return only 0 or 1, so int8 is okay.
  prediction = jax.nn.one_hot(
      jax.lax.convert_element_type(prediction, jnp.int8),
      num_classes=3,
      dtype=jnp.int8)
  prediction = prediction[..., 1:]

  if len(target.shape) >= 5:
    target = jnp.squeeze(target, axis=channel_axis)

  # As above, int8 is okay for one hot.
  target = jax.nn.one_hot(
      jax.lax.convert_element_type(target, jnp.int8),
      num_classes=3,
      dtype=jnp.int8)
  target = target[..., 1:]

  intersection = jnp.sum((target * prediction).astype(jnp.float32),
                         axis=reduce_axis)
  target_sum = jnp.sum(target.astype(jnp.float32), axis=reduce_axis)
  prediction_sum = jnp.sum(prediction.astype(jnp.float32), axis=reduce_axis)

  dice = (2.0 * intersection + smooth_nr) / (
      target_sum + prediction_sum + smooth_dr)
  return dice


def dice_score(prediction, target):
  """This is the special case of DiceScore from losses.py, separated for simplicity.

  Args:
    prediction: 5D numpy array. [Image, H, W, D, C].
    target: 4D [Image, H, W, D], or 5D [Image, H, W, D, C, 1] target classes
      array.

  Returns:
    Dice score per image, [B]. Does not computes mean unlike others.
  """
  smooth_nr = 1e-6
  smooth_dr = 1e-6
  channel_axis = -1
  reduce_axis = tuple(list(range(1, len(prediction.shape) - 1)))
  # arg max will simply return 0, 1, 2. int8 is okay to use.
  prediction = jax.lax.argmax(prediction,
                              len(prediction.shape) + channel_axis,
                              jnp.int8)
  if len(prediction.shape) >= 5:
    prediction = jnp.squeeze(prediction, axis=channel_axis)

  # One hot will also return only 0 or 1, so int8 is okay.
  prediction = jax.nn.one_hot(
      jax.lax.convert_element_type(prediction, jnp.int8),
      num_classes=3,
      dtype=jnp.int8)
  prediction = prediction[..., 1:]
  if len(target.shape) >= 5:
    target = jnp.squeeze(target, axis=channel_axis)

  # As above, int8 is okay for one hot.
  target = jax.nn.one_hot(
      jax.lax.convert_element_type(target, jnp.int8),
      num_classes=3,
      dtype=jnp.int8)

  target = target[..., 1:]

  intersection = jnp.sum((target * prediction).astype(jnp.float32),
                         axis=reduce_axis)
  target_sum = jnp.sum(target.astype(jnp.float32), axis=reduce_axis)
  prediction_sum = jnp.sum(prediction.astype(jnp.float32), axis=reduce_axis)

  dice = (2.0 * intersection + smooth_nr) / (
      target_sum + prediction_sum + smooth_dr)
  return dice


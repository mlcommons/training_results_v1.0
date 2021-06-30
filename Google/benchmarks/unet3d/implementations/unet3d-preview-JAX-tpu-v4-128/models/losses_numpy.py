"""NumPy implementation of losses in 3DUnet.

https://github.com/mmarcinkiewicz/training/blob/Add_unet3d/image_segmentation/unet3d/model/losses.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function


import numpy as np


def softmax(x):
  return np.exp(x) / np.exp(x).sum(-1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray,
                       one_hot_labels: np.ndarray) -> np.ndarray:
  """Returns the cross entropy loss between some logits and some labels.

  Args:
    logits: Output of the model.
    one_hot_labels: One-hot encoded labels. Dimensions should match the logits.

  Returns:
    The cross entropy, averaged over the first dimension (samples).
  """
  log_softmax_logits = np.log(softmax(logits))
  loss = -np.sum(one_hot_labels * log_softmax_logits, axis=-1)
  return np.mean(loss)


def compute_dice(prediction,
                 target,
                 to_onehot_y=True,
                 to_onehot_x=False,
                 use_softmax=True,
                 use_argmax=False,
                 include_background=False,
                 layout="NDHWC"):
  """Returns the dice coefficient between prediction and target.

  Args:
    prediction: Prediction.
    target: Target.
    to_onehot_y:
    to_onehot_x:
    use_softmax: Whether to use softmax.
    use_argmax: Whether to use argmax.
    include_background: Whether to include background.
    layout:

  Returns:
    The dice coefficient which is essentially a measure of overlap between two
    samples.
  """
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
    prediction = softmax(prediction)
  elif use_argmax:
    prediction = np.argmax(prediction, axis=channel_axis)

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

  intersection = np.sum(target * prediction, axis=reduce_axis)
  target_sum = np.sum(target, axis=reduce_axis)
  prediction_sum = np.sum(prediction, axis=reduce_axis)

  dice = (2.0 * intersection + smooth_nr) / (
      target_sum + prediction_sum + smooth_dr)
  return dice


def to_one_hot(array, layout, channel_axis):
  if len(array.shape) >= 5:
    array = np.squeeze(array, axis=channel_axis)
  array = np.array(array[..., np.newaxis] == np.arange(3), dtype=np.float32)
  if layout == "NCDHW":
    array = np.transpose(array, (0, 4, 1, 2, 3))
  return array


def compute_dice_ce_loss(y_pred,
                         y_true,
                         to_onehot_y,
                         use_softmax,
                         layout,
                         include_background=False):
  """Returns the average of the dice coeffcient and cross entropy.

  Args:
    y_pred: Prediction.
    y_true: Target.
    to_onehot_y:
    use_softmax: Whether to use softmax.
    layout:
    include_background: Whether to include background.

  Returns:
    The average of the dice coeffcient and cross entropy.
  """
  dice = 1.0 - np.mean(
      compute_dice(
          y_pred,
          y_true,
          to_onehot_y=to_onehot_y,
          use_softmax=use_softmax,
          include_background=include_background))
  if layout == "NCDHW":
    channel_axis = 1
  else:
    channel_axis = -1
  cross_entropy = cross_entropy_loss(y_pred,
                                     to_one_hot(y_true, layout, channel_axis))
  return (dice + cross_entropy) / 2


def compute_dice_score(y_pred,
                       y_true,
                       to_onehot_y=True,
                       use_argmax=True,
                       layout="NDHWC",
                       include_background=False,
                       compute_mean_score=True):
  """CPU compute dice score."""
  dice_scores = compute_dice(
      y_pred,
      y_true,
      to_onehot_y=to_onehot_y,
      to_onehot_x=True,
      use_softmax=False,
      use_argmax=use_argmax,
      layout=layout,
      include_background=include_background)
  if compute_mean_score:
    return np.mean(dice_scores, axis=0)
  else:
    return dice_scores

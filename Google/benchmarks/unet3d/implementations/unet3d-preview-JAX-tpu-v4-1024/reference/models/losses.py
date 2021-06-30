"""Reference implementation of losses.

https://github.com/mmarcinkiewicz/training/blame/Add_unet3d/image_segmentation/pytorch/model/losses.py
"""

import torch
import torch.nn as nn


def reduce_sum(x, axes, keepdim=False):
  # probably some check for uniqueness of axes
  if keepdim:
    for ax in axes:
      x = x.sum(ax, keepdim=True)
  else:
    for ax in sorted(axes, reverse=True):
      x = x.sum(ax)
  return x


class Dice:
  """Dice class."""

  def __init__(self,
               to_onehot_y: bool = True,
               to_onehot_x: bool = False,
               use_softmax: bool = True,
               use_argmax: bool = False,
               include_background: bool = False,
               layout: str = "NCDHW"):
    self.include_background = include_background
    self.to_onehot_y = to_onehot_y
    self.to_onehot_x = to_onehot_x
    self.use_softmax = use_softmax
    self.use_argmax = use_argmax
    self.smooth_nr = 1e-6
    self.smooth_dr = 1e-6
    self.layout = layout

  def __call__(self, prediction, target):
    if self.layout == "NCDHW":
      channel_axis = 1
      reduce_axis = list(range(2, len(prediction.shape)))
    else:
      channel_axis = -1
      reduce_axis = list(range(1, len(prediction.shape) - 1))
    num_pred_ch = prediction.shape[channel_axis]

    if self.use_softmax:
      # TODO: this line is equivalent implementation for
      # higher Pytorch version, check the correctness.
      prediction = nn.functional.softmax(prediction, dim=channel_axis)
    elif self.use_argmax:
      prediction = torch.argmax(prediction, dim=channel_axis)

    if self.to_onehot_y:
      target = to_one_hot(target, self.layout, channel_axis)

    if self.to_onehot_x:
      prediction = to_one_hot(prediction, self.layout, channel_axis)

    if not self.include_background:
      assert num_pred_ch > 1, \
          (f"To exclude background the prediction needs more than one channel. "
           f"Got {num_pred_ch}.")
      if self.layout == "NCDHW":
        target = target[:, 1:]
        prediction = prediction[:, 1:]
      else:
        target = target[..., 1:]
        prediction = prediction[..., 1:]

    assert (target.shape == prediction.shape), \
        (f"Target and prediction shape do not match. Target: ({target.shape}), "
         f"prediction: ({prediction.shape}).")

    # TODO: this line is equivalent implementation for
    # higher Pytorch version, check the correctness.
    intersection = reduce_sum(target * prediction, axes=reduce_axis)
    # TODO: this line is equivalent implementation for
    # higher Pytorch version, check the correctness.
    target_sum = reduce_sum(target, axes=reduce_axis)
    # TODO: this line is equivalent implementation for
    # higher Pytorch version, check the correctness.
    prediction_sum = reduce_sum(prediction, axes=reduce_axis)

    dice = (2.0 * intersection + self.smooth_nr) / (
        target_sum + prediction_sum + self.smooth_dr)
    return dice


def one_hot(y, num_classes):
  scatter_dim = len(y.size())
  y_tensor = y.view(*y.size(), -1)
  zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

  return zeros.scatter(scatter_dim, y_tensor, 1)


def to_one_hot(array, layout, channel_axis):
  if len(array.shape) >= 5:
    array = torch.squeeze(array, dim=channel_axis)
  # TODO: this line is equivalent implementation for
  # higher Pytorch version, check the correctness.
  array = one_hot(array.long(), num_classes=3)
  if layout == "NCDHW":
    array = array.permute(0, 4, 1, 2, 3).float()
  return array


class DiceCELoss(nn.Module):
  """DiceCELoss class."""

  def __init__(self,
               to_onehot_y,
               use_softmax,
               layout,
               include_background=False):
    super(DiceCELoss, self).__init__()
    self.dice = Dice(
        to_onehot_y=to_onehot_y,
        use_softmax=use_softmax,
        include_background=include_background)
    self.cross_entropy = nn.CrossEntropyLoss()

  def forward(self, y_pred, y_true):
    dice = 1.0 - torch.mean(self.dice(y_pred, y_true))
    cross_entropy = self.cross_entropy(y_pred,
                                       torch.squeeze(y_true, dim=1).long())
    return (dice + cross_entropy) / 2


class DiceScore:
  """DiceScore class."""

  def __init__(self,
               to_onehot_y: bool = True,
               use_argmax: bool = True,
               layout: str = "NCDHW",
               include_background: bool = False):
    self.dice = Dice(
        to_onehot_y=to_onehot_y,
        to_onehot_x=True,
        use_softmax=False,
        use_argmax=use_argmax,
        layout=layout,
        include_background=include_background)

  def __call__(self, y_pred, y_true):
    return torch.mean(self.dice(y_pred, y_true), dim=0)

"""JAX implementation of layers in 3DUnet.

Reference:
https://github.com/mmarcinkiewicz/training/blob/Add_unet3d/image_segmentation/unet3d/models/layers.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

import functools
from absl import flags

from flax import nn
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

flags.DEFINE_bool(
    "instance_norm_propogate_bf16",
    help="BF16 propogation in instance norm",
    default=False)

flags.DEFINE_bool(
    "enable_concat_upsample",
    help="If true conv(x+skip), otherwise conv(x) + conv(skip).",
    default=True)

FLAGS = flags.FLAGS


def ignore_print(tensor_name):
  del tensor_name


class Identity(nn.Module):

  def apply(self, x):
    return x


class InstanceNorm(nn.Module):
  """Instance normalization (https://arxiv.org/abs/1607.08022).

  Operates on the batch and channel axes of the input data.
  """

  def apply(self,
            x,
            axis=(1, 2, 3),
            epsilon=1e-5,
            bias=True,
            scale=True,
            bias_init=initializers.zeros,
            scale_init=initializers.ones):
    """Applies instance normalization on the input.

    It normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Args:
      x: the inputs
      axis: the batch and channel axis of the input.
      epsilon: A small float added to variance to avoid dividing by zero.
      bias:  If True, bias (beta) is added.
      scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.

    Returns:
      Normalized inputs (the same shape as inputs).

    """
    dtype = x.dtype
    # Normalization might need more testing. So, if bf16 set,
    # normalization is done only if instance_norm_propogate_bf16=True.
    if FLAGS.instance_norm_propogate_bf16:
      intermediate_dtype = dtype
    else:
      intermediate_dtype = jnp.float32
    if not FLAGS.instance_norm_propogate_bf16:
      x = jnp.asarray(x, jnp.float32)
    assert x.dtype == intermediate_dtype
    features = x.shape[-1]
    mean = jnp.mean(x, axis=axis, keepdims=True)
    assert mean.dtype == intermediate_dtype
    var = jnp.mean(lax.square(x - mean), axis=axis, keepdims=True)
    assert var.dtype == intermediate_dtype
    mul = lax.rsqrt(var + epsilon)
    assert mul.dtype == intermediate_dtype
    if scale:
      if FLAGS.instance_norm_propogate_bf16:
        mul = mul * jnp.asarray(
            self.param("scale", (features,), scale_init), jnp.float32).astype(
                dtype)
      else:
        mul = mul * jnp.asarray(
            self.param("scale", (features,), scale_init), jnp.float32)
    assert mul.dtype == intermediate_dtype
    y = (x - mean) * mul
    assert y.dtype == intermediate_dtype
    if bias:
      if FLAGS.instance_norm_propogate_bf16:
        y = y + jnp.asarray(self.param("bias", (features,), bias_init),
                            jnp.float32).astype(dtype)
      else:
        y = y + jnp.asarray(self.param("bias", (features,), bias_init),
                            jnp.float32)
    assert y.dtype == intermediate_dtype
    return jnp.asarray(y, dtype)


activations = {
    "relu": nn.relu,
    "leaky_relu": nn.leaky_relu,
    "sigmoid": nn.sigmoid,
    "softmax": functools.partial(nn.softmax, axis=1),
    "none": lambda x: x,
}

normalizations = {
    "instancenorm": InstanceNorm,
    "batchnorm": nn.BatchNorm,
    "none": Identity,
}

convolutions = {"transpose": nn.ConvTranspose, "regular": nn.Conv}


def _normalization(normalization):
  if normalization in normalizations:
    return normalizations[normalization]
  raise ValueError(f"Unknown normalization {normalization}")


def _activation(activation):
  if activation in activations:
    return activations[activation]
  raise ValueError(f"Unknown activation {activation}")


class ConvBlockFactory(nn.Module):
  """convolution factory."""

  def apply(self,
            x,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_type="regular",
            normalization="instancenorm",
            activation="leaky_relu",
            tensor_name="",
            print_func=ignore_print):
    dtype = x.dtype
    conv = convolutions[conv_type]
    conv = conv.partial(
        features=out_channels,
        kernel_size=(kernel_size,) * (x.ndim - 2),
        strides=(stride,) * (x.ndim - 2),
        padding=((padding, padding),) * (x.ndim - 2),
        bias=normalization == "none",
        dtype=dtype)

    norm_layer = _normalization(normalization)
    activation = _activation(activation)
    x = conv(x)
    print_func(tensor_name + "conv")
    assert x.dtype == dtype
    x = norm_layer(x)
    if normalization != "none":
      print_func(tensor_name + normalization)
    assert x.dtype == dtype
    x = activation(x)
    assert x.dtype == dtype
    return x


class DownsampleBlock(nn.Module):
  """DownsampleBlock."""

  def apply(self, x, in_channels, out_channels, normalization, activation,
            tensor_name="", print_func=ignore_print):
    dtype = x.dtype
    conv1 = ConvBlockFactory.partial(
        in_channels=in_channels,
        out_channels=out_channels,
        normalization=normalization,
        stride=2,
        activation=activation,
        tensor_name=tensor_name + "_block_0_",
        print_func=print_func)
    conv2 = ConvBlockFactory.partial(
        in_channels=out_channels,
        out_channels=out_channels,
        normalization=normalization,
        activation=activation,
        tensor_name=tensor_name + "_block_1_",
        print_func=print_func)
    x = conv1(x)
    assert x.dtype == dtype
    x = conv2(x)
    assert x.dtype == dtype
    return x


class ConvBlockFactoryWithoutConcat(nn.Module):
  """convolution factory."""

  def apply(self,
            x, skip,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_type="regular",
            normalization="instancenorm",
            activation="leaky_relu"):
    assert normalization != "none"
    assert conv_type == "regular"

    dtype = x.dtype
    conv = convolutions[conv_type]
    conv1 = conv.partial(
        features=out_channels,
        kernel_size=(kernel_size,) * (x.ndim - 2),
        strides=(stride,) * (x.ndim - 2),
        padding=((padding, padding),) * (x.ndim - 2),
        bias=normalization == "none",
        dtype=dtype)
    conv2 = conv.partial(
        features=out_channels,
        kernel_size=(kernel_size,) * (x.ndim - 2),
        strides=(stride,) * (x.ndim - 2),
        padding=((padding, padding),) * (x.ndim - 2),
        bias=normalization == "none",
        dtype=dtype)
    normalization = _normalization(normalization)
    activation = _activation(activation)
    x = conv1(x) + conv2(skip)
    assert x.dtype == dtype
    x = normalization(x)
    assert x.dtype == dtype
    x = activation(x)
    assert x.dtype == dtype
    return x


class UpsampleBlock(nn.Module):
  """UnsampleBlock."""

  def apply(self, x, skip, in_channels, out_channels, normalization,
            activation, tensor_name="", print_func=ignore_print):
    dtype = x.dtype
    assert skip.dtype == dtype
    upsample_conv = ConvBlockFactory.partial(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=2,
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html
        # real padding added to transpose conv3D is:
        # dilation * (kernel_size - 1) - padding
        # Pytorch reference use padding = 0, (dilation * kernel_size - 1) = 1
        padding=1,
        conv_type="transpose",
        normalization="none",
        activation="none",
        tensor_name=tensor_name + "_de",
        print_func=print_func)

    conv1 = ConvBlockFactory.partial(
        in_channels=2 * out_channels,
        out_channels=out_channels,
        normalization=normalization,
        activation=activation,
        tensor_name=tensor_name + "_block_0_",
        print_func=print_func)
    conv2 = ConvBlockFactory.partial(
        in_channels=out_channels,
        out_channels=out_channels,
        normalization=normalization,
        activation=activation,
        tensor_name=tensor_name + "_block_1_",
        print_func=print_func)
    x = upsample_conv(x)
    assert x.dtype == dtype
    # pytorch reference use data format (N,C,D,H,W)
    # JAX implementation use data format (N,D,H,W,C)
    if FLAGS.enable_concat_upsample:
      x = jnp.concatenate((x, skip), axis=4)
      assert x.dtype == dtype
      x = conv1(x)
    else:
      conv_with_concat = ConvBlockFactoryWithoutConcat.partial(
          in_channels=2 * out_channels,
          out_channels=out_channels,
          normalization=normalization,
          activation=activation)
      x = conv_with_concat(x, skip)
    assert x.dtype == dtype
    x = conv2(x)
    assert x.dtype == dtype
    return x


class InputBlock(nn.Module):
  """InputBlock."""

  def apply(self, x, in_channels, out_channels, normalization, activation,
            tensor_name="", print_func=ignore_print):
    dtype = x.dtype
    conv1 = ConvBlockFactory.partial(
        in_channels=in_channels,
        out_channels=out_channels,
        normalization=normalization,
        activation=activation,
        tensor_name=tensor_name + "_block_0_",
        print_func=print_func)
    conv2 = ConvBlockFactory.partial(
        in_channels=out_channels,
        out_channels=out_channels,
        normalization=normalization,
        activation=activation,
        tensor_name=tensor_name + "_block_1_",
        print_func=print_func)
    x = conv1(x)
    assert x.dtype == dtype
    x = conv2(x)
    assert x.dtype == dtype
    return x


class OutputLayer(nn.Module):
  """OutputLayer."""

  def apply(self, x, in_channels, n_class,
            tensor_name="", print_func=ignore_print):
    dtype = x.dtype
    conv = ConvBlockFactory.partial(
        in_channels=in_channels,
        out_channels=n_class,
        kernel_size=1,
        padding=0,
        activation="none",
        normalization="none",
        tensor_name=tensor_name + "_",
        print_func=print_func)
    x = conv(x)
    assert x.dtype == dtype
    return x

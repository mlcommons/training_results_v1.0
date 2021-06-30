"""JAX implementation of unet3d.

https://github.com/mmarcinkiewicz/training/blob/Add_unet3d/image_segmentation/unet3d/model/losses.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

from flax import nn
import jax
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.models import layers

# pylint: disable=g-complex-comprehension


class Unet3D(nn.Module):
  """Unet3D class."""

  def apply(self, x, in_channels, n_class, normalization, activation,
            print_func=layers.ignore_print):
    filters = [32, 64, 128, 256, 320]

    inp = filters[:-1]
    out = filters[1:]
    input_dim = filters[0]

    input_block = layers.InputBlock.partial(
        in_channels=in_channels,
        out_channels=input_dim,
        normalization=normalization,
        activation=activation,
        print_func=print_func)

    downsample_fns = [
        layers.DownsampleBlock.partial(
            in_channels=i,
            out_channels=o,
            normalization=normalization,
            activation=activation,
            print_func=print_func) for i, o in zip(inp, out)
    ]
    bottleneck = layers.DownsampleBlock.partial(
        in_channels=filters[-1],
        out_channels=filters[-1],
        normalization=normalization,
        activation=activation,
        print_func=print_func)
    upsample_fns = [
        layers.UpsampleBlock.partial(
            in_channels=filters[-1],
            out_channels=filters[-1],
            normalization=normalization,
            activation=activation,
            print_func=print_func)
    ]
    upsample_fns.extend([
        layers.UpsampleBlock.partial(
            in_channels=i,
            out_channels=o,
            normalization=normalization,
            activation=activation,
            print_func=print_func)
        for i, o in zip(reversed(out), reversed(inp))
    ])
    output = layers.OutputLayer.partial(in_channels=input_dim, n_class=n_class,
                                        print_func=print_func)
    # Introduce no-op jitted functions, so that profiler can show them in stacks
    # pylint: disable=unnecessary-lambda,cell-var-from-loop
    @jax.jit
    def jinput_block(y):
      return input_block(y, tensor_name="input")

    x = jinput_block(x)
    outputs = [x]

    down_index = 0
    for downsample in downsample_fns:
      @jax.jit
      def jdownsample(y):
        return downsample(y, tensor_name="down%s" % down_index)
      x = jdownsample(x)
      down_index += 1
      outputs.append(x)

    @jax.jit
    def jbottleneck(y):
      return bottleneck(y, tensor_name="down%s" % down_index)

    x = jbottleneck(x)
    up_index = 0
    for upsample, skip in zip(upsample_fns, reversed(outputs)):
      @jax.jit
      def jupsample(y, z):
        return upsample(y, z, tensor_name="up%s" % up_index)
      x = jupsample(x, skip)
      up_index += 1

    @jax.jit
    def joutput_block(y):
      return output(y, tensor_name="output")
    x = joutput_block(x)
    # pylint: enable=unnecessary-lambda,cell-var-from-loop

    return x

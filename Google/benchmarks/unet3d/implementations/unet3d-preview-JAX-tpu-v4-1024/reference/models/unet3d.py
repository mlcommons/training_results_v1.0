"""Reference implementation of unet3d.

https://github.com/mmarcinkiewicz/training/blob/Add_unet3d/image_segmentation/unet3d/model/losses.py
"""

import torch.nn as nn

from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.reference.models.layers import DownsampleBlock
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.reference.models.layers import InputBlock
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.reference.models.layers import OutputLayer
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.reference.models.layers import UpsampleBlock


class Unet3D(nn.Module):
  """Unet3D class."""

  def __init__(self, in_channels, n_class, normalization, activation):
    super(Unet3D, self).__init__()

    filters = [32, 64, 128, 256, 320]
    self.filters = filters

    self.inp = filters[:-1]
    self.out = filters[1:]
    input_dim = filters[0]

    self.input_block = InputBlock(in_channels, input_dim, normalization,
                                  activation)

    self.downsample = nn.ModuleList([
        DownsampleBlock(i, o, normalization, activation)
        for i, o in zip(self.inp, self.out)
    ])
    self.bottleneck = DownsampleBlock(filters[-1], filters[-1], normalization,
                                      activation)
    upsample = [
        UpsampleBlock(filters[-1], filters[-1], normalization, activation)
    ]
    upsample.extend([
        UpsampleBlock(i, o, normalization, activation)
        for i, o in zip(reversed(self.out), reversed(self.inp))
    ])
    self.upsample = nn.ModuleList(upsample)
    self.output = OutputLayer(input_dim, n_class)

  def forward(self, x):
    x = self.input_block(x)
    outputs = [x]

    for downsample in self.downsample:
      x = downsample(x)
      outputs.append(x)

    x = self.bottleneck(x)

    for upsample, skip in zip(self.upsample, reversed(outputs)):
      x = upsample(x, skip)

    x = self.output(x)

    return x

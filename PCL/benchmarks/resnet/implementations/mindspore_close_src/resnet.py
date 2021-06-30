# Copyright 2021 PCL & PKU
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""ResNet."""
import numpy as np
from functools import partial

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from scipy.stats import truncnorm
from mindspore.nn import GlobalBatchNorm
from mindspore.common.initializer import HeUniform, HeNormal, XavierUniform


def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale**0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


class LayerBuilder(object):
    def __init__(self, conv_init_mode='truncnorm', bn_init_mode='adv_bn_init',
                 syncbn_idxs=(), syncbn_group_size=2):
        assert conv_init_mode in ['truncnorm', 'HeUniform', 'XavierUniform', 'HeNormal']
        assert bn_init_mode in ['adv_bn_init', 'conv_bn_init']

        # conv
        self.conv_init_mode = conv_init_mode

        # batchnorm
        self.bn_init_mode = bn_init_mode
        self.bn_eps = 1e-5
        self.bn_momentum = 0.9

    def conv2d(self, in_channel, out_channel, kernel, stride=1):
        if self.conv_init_mode == 'truncnorm':
            weight = _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=kernel)
        elif self.conv_init_mode == 'HeNormal':
            weight = HeNormal(mode='fan_out', nonlinearity='relu')
        elif self.conv_init_mode == 'HeUniform':
            weight = 'HeUniform'
        elif self.conv_init_mode == 'XavierUniform':
            raise NotImplementedError

        conv_op = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride,
                            padding=0, pad_mode='same', weight_init=weight)
        return conv_op

    def batchnorm2d(self, channel, is_last=False):
        gamma_init = 0 if is_last and self.bn_init_mode == 'adv_bn_init' else 1
        bn_op = nn.BatchNorm2d(channel, eps=self.bn_eps, momentum=self.bn_momentum,
                        gamma_init=gamma_init, beta_init=0, moving_mean_init=0, moving_var_init=1)

        return bn_op

    def fc(self, in_channel, out_channel):
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
        fc_op = nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)
        return fc_op


class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 builder,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = builder.conv2d(in_channel, channel, 1, stride=1)
        self.bn1 = builder.batchnorm2d(channel)

        self.conv2 = builder.conv2d(channel, channel, 3, stride=stride)
        self.bn2 = builder.batchnorm2d(channel)

        self.conv3 = builder.conv2d(channel, out_channel, 1, stride=1)
        self.bn3 = builder.batchnorm2d(out_channel, is_last=True)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([
                                        builder.conv2d(in_channel, out_channel, 1, stride),
                                        builder.batchnorm2d(out_channel)])
        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 conv_init_mode='truncnorm',
                 bn_init_mode='adv_bn_init'):

        self.builder = LayerBuilder(conv_init_mode=conv_init_mode,
                                    bn_init_mode=bn_init_mode)

        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = self.builder.conv2d(3, 64, 7, stride=2)
        self.bn1 = self.builder.batchnorm2d(64)

        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = self.builder.fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.

        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        layers = []

        resnet_block = block(self.builder, in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(self.builder, out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def resnet50(backbone='resnet50',
             class_num=10,  
             conv_init_mode='truncnorm',
             bn_init_mode='adv_bn_init'):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50(10)
    """

    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  conv_init_mode=conv_init_mode,
                  bn_init_mode=bn_init_mode)

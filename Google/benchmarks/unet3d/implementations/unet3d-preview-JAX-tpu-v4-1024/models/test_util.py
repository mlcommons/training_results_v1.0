"""Test utilities."""

from absl import flags
import numpy as np
import torch

from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.models import layers  # pylint: disable=unused-import

FLAGS = flags.FLAGS


def jax_param_to_torch(x):
  return torch.nn.Parameter(torch.from_numpy(np.array(x)))


def convert_down_sample_weight_to_torch(jax_down_sample_params,
                                        torch_down_params):
  """Converts down sample jax weights to torch."""
  conv0_weight = np.transpose(
      jax_down_sample_params['ConvBlockFactory_0']['Conv_0']['kernel'],
      [4, 3, 0, 1, 2])
  torch_down_params.conv1[0].weight = jax_param_to_torch(conv0_weight)

  conv1_weight = np.transpose(
      jax_down_sample_params['ConvBlockFactory_1']['Conv_0']['kernel'],
      [4, 3, 0, 1, 2])
  torch_down_params.conv2[0].weight = jax_param_to_torch(conv1_weight)

  torch_down_params.conv1[1].weight = jax_param_to_torch(
      jax_down_sample_params['ConvBlockFactory_0']['InstanceNorm_1']['scale'])
  torch_down_params.conv1[1].bias = jax_param_to_torch(
      jax_down_sample_params['ConvBlockFactory_0']['InstanceNorm_1']['bias'])

  torch_down_params.conv2[1].weight = jax_param_to_torch(
      jax_down_sample_params['ConvBlockFactory_1']['InstanceNorm_1']['scale'])
  torch_down_params.conv2[1].bias = jax_param_to_torch(
      jax_down_sample_params['ConvBlockFactory_1']['InstanceNorm_1']['bias'])
  return torch_down_params


def convert_input_block_to_torch(jax_params, torch_params):
  """Converts input block jax weights to torch."""
  conv2_weight = np.transpose(
      jax_params['ConvBlockFactory_0']['Conv_0']['kernel'], [4, 3, 0, 1, 2])
  torch_params.conv1[0].weight = jax_param_to_torch(conv2_weight)

  torch_params.conv1[1].weight = jax_param_to_torch(
      jax_params['ConvBlockFactory_0']['InstanceNorm_1']['scale'])
  torch_params.conv1[1].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_0']['InstanceNorm_1']['bias'])

  conv2_weight = np.transpose(
      jax_params['ConvBlockFactory_1']['Conv_0']['kernel'], [4, 3, 0, 1, 2])
  torch_params.conv2[0].weight = jax_param_to_torch(conv2_weight)

  torch_params.conv2[1].weight = jax_param_to_torch(
      jax_params['ConvBlockFactory_1']['InstanceNorm_1']['scale'])
  torch_params.conv2[1].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_1']['InstanceNorm_1']['bias'])
  return torch_params


def convert_upsample_block_to_torch_with_concat(jax_params, torch_params):
  """Converts upsample jax weights to torch."""
  kernel = jax_params['ConvBlockFactory_0']['ConvTranspose_0']['kernel']
  # Flax ConvTranspose3D does not do the flip, for numerical equivalance,
  # flip kernel.
  kernel = np.flip(kernel, [0, 1, 2])
  # (2, 2, 2, 2, 8) -> ([2, 8, 2, 2, 2])
  conv0_weight = np.transpose(kernel, [3, 4, 0, 1, 2])
  torch_params.upsample_conv[0].weight = jax_param_to_torch(conv0_weight)
  torch_params.upsample_conv[0].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_0']['ConvTranspose_0']['bias'])

  # (3, 3, 3, 16, 8) -> ([8, 16, 3, 3, 3])
  conv1_weight = np.transpose(
      jax_params['ConvBlockFactory_1']['Conv_0']['kernel'], [4, 3, 0, 1, 2])
  torch_params.conv1[0].weight = jax_param_to_torch(conv1_weight)

  torch_params.conv1[1].weight = jax_param_to_torch(
      jax_params['ConvBlockFactory_1']['InstanceNorm_1']['scale'])
  torch_params.conv1[1].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_1']['InstanceNorm_1']['bias'])

  # (3, 3, 3, 8, 8) -> ([8, 8, 3, 3, 3])
  conv2_weight = np.transpose(
      jax_params['ConvBlockFactory_2']['Conv_0']['kernel'], [4, 3, 0, 1, 2])
  torch_params.conv2[0].weight = jax_param_to_torch(conv2_weight)

  torch_params.conv2[1].weight = jax_param_to_torch(
      jax_params['ConvBlockFactory_2']['InstanceNorm_1']['scale'])
  torch_params.conv2[1].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_2']['InstanceNorm_1']['bias'])
  return torch_params


def convert_upsample_block_to_torch_without_concat(jax_params, torch_params):
  """Converts upsample jax weights to torch."""
  kernel = jax_params['ConvBlockFactory_0']['ConvTranspose_0']['kernel']
  # Flax ConvTranspose3D does not do the flip, for numerical equivalance,
  # flip kernel.
  kernel = np.flip(kernel, [0, 1, 2])
  # (2, 2, 2, 2, 8) -> ([2, 8, 2, 2, 2])
  conv0_weight = np.transpose(kernel, [3, 4, 0, 1, 2])
  torch_params.upsample_conv[0].weight = jax_param_to_torch(conv0_weight)
  torch_params.upsample_conv[0].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_0']['ConvTranspose_0']['bias'])

  # (3, 3, 3, 8, 8) -> ([8, 8, 3, 3, 3])
  conv1_weight_1 = np.transpose(
      jax_params['ConvBlockFactoryWithoutConcat_1']['Conv_0']['kernel'],
      [4, 3, 0, 1, 2])
  # (3, 3, 3, 8, 8) -> ([8, 8, 3, 3, 3])
  conv1_weight_2 = np.transpose(
      jax_params['ConvBlockFactoryWithoutConcat_1']['Conv_1']['kernel'],
      [4, 3, 0, 1, 2])
  conv1_weight = np.concatenate([conv1_weight_1, conv1_weight_2], axis=1)

  torch_params.conv1[0].weight = jax_param_to_torch(conv1_weight)
  torch_params.conv1[1].weight = jax_param_to_torch(
      jax_params['ConvBlockFactoryWithoutConcat_1']['InstanceNorm_2']['scale'])
  torch_params.conv1[1].bias = jax_param_to_torch(
      jax_params['ConvBlockFactoryWithoutConcat_1']['InstanceNorm_2']['bias'])

  # (3, 3, 3, 8, 8) -> ([8, 8, 3, 3, 3])
  conv2_weight = np.transpose(
      jax_params['ConvBlockFactory_2']['Conv_0']['kernel'], [4, 3, 0, 1, 2])
  torch_params.conv2[0].weight = jax_param_to_torch(conv2_weight)

  torch_params.conv2[1].weight = jax_param_to_torch(
      jax_params['ConvBlockFactory_2']['InstanceNorm_1']['scale'])
  torch_params.conv2[1].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_2']['InstanceNorm_1']['bias'])
  return torch_params


def convert_upsample_block_to_torch(jax_params, torch_params):
  if FLAGS.enable_concat_upsample:
    return convert_upsample_block_to_torch_with_concat(jax_params, torch_params)
  else:
    return convert_upsample_block_to_torch_without_concat(jax_params,
                                                          torch_params)


def convert_output_block_to_torch(jax_params, torch_params):
  """Converts output block jax weights to torch."""
  conv2_weight = np.transpose(
      jax_params['ConvBlockFactory_0']['Conv_0']['kernel'], [4, 3, 0, 1, 2])
  torch_params.conv[0].weight = jax_param_to_torch(conv2_weight)
  torch_params.conv[0].bias = jax_param_to_torch(
      jax_params['ConvBlockFactory_0']['Conv_0']['bias'])
  return torch_params


def convert_jax_unet3d_weights_to_torch(jax_params, pytorch_model):
  """Converts JAX unet3d weights to torch."""
  downblocknames = [
      k for k in jax_params.keys() if k.startswith('DownsampleBlock_')
  ]

  downblocknames = sorted(
      downblocknames, key=lambda x: int(x.split('DownsampleBlock_')[-1]))
  bottleneck_layer = downblocknames[-1]

  for i, downblock in enumerate(downblocknames[:-1]):
    pytorch_model.downsample[i] = convert_down_sample_weight_to_torch(
        jax_params[downblock], pytorch_model.downsample[i])
  pytorch_model.bottleneck = convert_down_sample_weight_to_torch(
      jax_params[bottleneck_layer], pytorch_model.bottleneck)

  convert_input_block_to_torch(jax_params['InputBlock_0'],
                               pytorch_model.input_block)

  upblocknames = [
      k for k in jax_params.keys() if k.startswith('UpsampleBlock_')
  ]
  upblocknames = sorted(
      upblocknames, key=lambda x: int(x.split('UpsampleBlock_')[-1]))

  for i, upblockname in enumerate(upblocknames):
    pytorch_model.upsample[i] = convert_upsample_block_to_torch(
        jax_params[upblockname], pytorch_model.upsample[i])

  pytorch_model.output = convert_output_block_to_torch(
      jax_params['OutputLayer_11'], pytorch_model.output)
  return pytorch_model


"""Guassian kernel implementation from reference model."""

import numpy as np
from scipy import signal


def gaussian_kernel(n, std):
  """Gaussian kernel."""
  # pylint: disable=invalid-name
  gaussian1D = signal.gaussian(n, std)
  gaussian2D = np.outer(gaussian1D, gaussian1D)
  gaussian3D = np.outer(gaussian2D, gaussian1D)
  gaussian3D = gaussian3D.reshape(n, n, n)
  gaussian3D = np.cbrt(gaussian3D)
  gaussian3D /= gaussian3D.max()
  return gaussian3D
  # pylint: enable=invalid-name


def get_norm_patch(params):
  """Creates the normalization patch tensor."""
  eval_mode = params["eval_mode"]
  roi_shape = params["val_input_shape_without_channel"]
  if eval_mode == "constant":
    norm_patch = np.ones(shape=roi_shape, dtype=np.float32)
  elif eval_mode == "gaussian":
    norm_patch = gaussian_kernel(roi_shape[0],
                                 0.125 * roi_shape[0]).astype(np.float32)
    norm_patch = np.array(norm_patch)
  else:
    raise ValueError("Unknown mode. Available modes are {constant, gaussian}.")
  norm_patch = np.reshape(norm_patch, norm_patch.shape + (1,))
  return norm_patch

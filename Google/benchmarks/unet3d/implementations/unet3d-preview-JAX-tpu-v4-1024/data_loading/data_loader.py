"""Pytorch reference for data_loader.

https://github.com/mmarcinkiewicz/training/blob/Add_unet3d/image_segmentation/pytorch/data_loading/data_loader.py
"""

import os

import numpy as np
import tensorflow.compat.v1 as tf
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.data_loading import input_reader


def load_data(path, files_pattern):
  data = sorted(tf.io.gfile.glob(os.path.join(path, files_pattern)))
  return data


def get_split(data, train_idx, val_idx):
  train = list(np.array(data)[train_idx])
  val = list(np.array(data)[val_idx])
  return train, val


def get_data_split(path, params):
  """Get data split for training and eval."""
  val_cases_list = params["eval_image_indices"]
  tfrecords = load_data(path, "*.tfrecord")
  data_train, data_val = [], []
  for tfrecord in tfrecords:
    # tfrecord is similar to
    # /REDACTED/nm-d/home/tpu-perf-team/unet3d/kits19/numpy_data/case_00003.tfrecord
    # val_cases_list contains only the 5 digit number.
    if tfrecord.split("_")[-1].split(".")[0] in val_cases_list:
      data_val.append(tfrecord)
    else:
      data_train.append(tfrecord)
  return data_train, data_val


def get_data_loaders(data_dir, params):
  """Get data loaders for training and eval."""

  data_train, data_val = get_data_split(data_dir, params)
  train_params = params
  train_params["is_training"] = True
  use_fake_data = params.get("use_fake_data", False)
  use_fake_train_data = params.get("use_fake_train_data", False)
  if use_fake_data or use_fake_train_data:
    train_dataloader = input_reader.FakeDatasetFn(train_params)
  else:
    train_dataloader = input_reader.InputFn(data_train, train_params)

  eval_params = params
  eval_params["is_training"] = False

  if use_fake_data:
    eval_dataloader = input_reader.FakeDatasetFn(eval_params)
  else:
    eval_dataloader = input_reader.InputFn(data_val, eval_params)

  return train_dataloader, eval_dataloader

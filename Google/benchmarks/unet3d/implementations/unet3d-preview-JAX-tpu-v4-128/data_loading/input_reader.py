"""input_reader for 3D Unet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import REDACTED
from __future__ import print_function

import functools
import math
import random

import numpy as np
import scipy.ndimage
import tensorflow.compat.v1 as tf
from REDACTED.mlperf.submissions.training.v1_0.models.unet3d.helpers import helpers


NUM_CHANNELS = 1
# Max image dimensions after processed for sliding windows
MAX_EVAL_IMAGE_SHAPE = np.array([448, 448, 448, NUM_CHANNELS],
                                dtype=np.int64)

# Max raw image dimensions stored before processed for sliding windows
MAX_RAW_EVAL_IMAGE_SHAPE = np.array([456, 416, 416, NUM_CHANNELS],
                                    dtype=np.int64)

NUM_EVAL_SAMPLES = 42


def _slice_image_for_testing(image, label, image_shape, params):
  """For testing, create smaller images."""
  test_dataset_shape = params.get('test_dataset_shape', None)
  if test_dataset_shape:
    assert len(test_dataset_shape) == 4
    test_dataset_dynamism = params.get('test_dataset_dynamism', 0)
    if test_dataset_dynamism:
      dynamic_shape = list(np.random.randint(-test_dataset_dynamism,
                                             test_dataset_dynamism, [3]))
      dynamic_shape.append(0)
      test_dataset_shape = np.array(dynamic_shape) + np.array(
          test_dataset_shape)

    image_shape = tf.minimum(test_dataset_shape, image_shape)
    begin = tf.constant([0, 0, 0, 0], dtype=image_shape.dtype)
    image = tf.slice(image, begin, image_shape)
    label = tf.slice(label, begin, image_shape)
  return image, label, image_shape


def _move_image_channels_to_last_dim(image, label, image_shape):
  """Moves the channel dim to last dimension in the images and labels."""
  # TODO: Move below transformation to training dataset as well.
  # The images are stored so that they have [C, H, W, D].
  # Since C appears to be always 1, we simply reshape it without transpose
  # to make it [H, W, D, 1]
  expected_channels = tf.constant(NUM_CHANNELS, dtype=tf.int64)
  single_channel_image_check = tf.assert_equal(image_shape[0],
                                               expected_channels)
  with tf.control_dependencies([single_channel_image_check]):
    image_shape = tf.concat([image_shape[1:], [expected_channels]], axis=0)
    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, image_shape)
  return image, label, image_shape


def get_norm_map_fn(params):
  """Returns a function that creates norm_maps for the given image size."""
  def make_norm_map_for_image(image_shape, stride_sizes):
    overlap = params['overlap']  # 0.5
    roi_shape = params['val_input_shape_without_channel']  # [128, 128, 128]
    strides = (np.array(roi_shape) * (1 - overlap)).astype(np.int64)
    norm_map = np.zeros(image_shape, dtype=np.float32)
    patch = helpers.get_norm_patch(params)
    for i in range(0, strides[0] * stride_sizes[0], strides[0]):
      for j in range(0, strides[1] * stride_sizes[1], strides[1]):
        for k in range(0, strides[2] * stride_sizes[2], strides[2]):
          norm_map[
              i:(roi_shape[0] + i),
              j:(roi_shape[1] + j),
              k:(roi_shape[2] + k), :] += patch
    return norm_map
  return make_norm_map_for_image


def make_sliding_windows(raw_image, raw_label, image_roi, left_padding,
                         right_padding, stride_sizes, image_idx, params):
  """Converts an image to its sliding windows."""
  roi_shape = params['val_input_shape_without_channel']  # [128, 128, 128, 1]
  roi_shape_5d = [1] + roi_shape + [1]
  overlap = params['overlap']  # 0.5
  # 64, 64, 64, 0
  strides = (np.array(roi_shape) * (1 - overlap)).astype(np.int64)
  strides_5d = [1] + list(strides) + [1]

  norm_map = tf.py_function(func=get_norm_map_fn(params),
                            inp=[image_roi, stride_sizes],
                            Tout=tf.float32)
  if params.get('use_bfloat16', False):
    norm_map = tf.cast(norm_map, tf.bfloat16)

  norm_maps = tf.extract_volume_patches(
      tf.expand_dims(norm_map, axis=0),
      roi_shape_5d, strides_5d, padding='VALID'
  )
  norm_maps = tf.reshape(norm_maps, [-1] + roi_shape + [1])

  images = tf.extract_volume_patches(
      tf.expand_dims(raw_image, axis=0),
      roi_shape_5d, strides_5d, padding='VALID'
  )
  images = tf.reshape(images, [-1] + roi_shape + [1])

  labels = tf.extract_volume_patches(
      tf.expand_dims(raw_label, axis=0),
      roi_shape_5d, strides_5d, padding='VALID'
  )
  labels = tf.reshape(labels, [-1] + roi_shape + [1])

  mask = tf.ones(
      [image_roi[0] - right_padding[0] - left_padding[0],
       image_roi[1] - right_padding[1] - left_padding[1],
       image_roi[2] - right_padding[2] - left_padding[2],
       1], dtype=tf.int8)
  mask = tf.pad(
      mask,
      [[left_padding[0], right_padding[0]], [left_padding[1], right_padding[1]],
       [left_padding[2], right_padding[2]], [0, 0]],
      mode='constant')

  masks = tf.extract_volume_patches(
      tf.expand_dims(mask, axis=0),
      roi_shape_5d, strides_5d, padding='VALID'
  )
  masks = tf.reshape(masks, [-1] + roi_shape + [1])

  num_windows = tf.math.reduce_prod(stride_sizes)
  is_padding = tf.zeros([num_windows], dtype=tf.bool)
  image_indices = tf.ones([num_windows], dtype=tf.int64) * image_idx

  window_indices = tf.cast(tf.range(num_windows), tf.int64)
  crop_locs = tf.transpose(
      tf.stack([(window_indices // stride_sizes[2]) // stride_sizes[1],
                (window_indices // stride_sizes[2]) % stride_sizes[1],
                window_indices % stride_sizes[2],
                tf.zeros([num_windows], dtype=tf.int64)]))
  crop_locs = crop_locs * strides_5d[1:]
  crop_locs = tf.cast(crop_locs, tf.int32)
  image_indices = tf.cast(image_indices, tf.int32)
  return images, labels, crop_locs, image_indices, masks, is_padding, norm_maps


def prepare_sliding_window(image, label, image_idx, image_shape, params):
  """Given an unpadded image it pads it to max eval shape, and creates sliding windows."""
  # No need to store this large, as it is 0, 1 or 2.
  label = tf.cast(label, tf.int8)
  if params.get('use_bfloat16', False):
    image = tf.cast(image, tf.bfloat16)

  overlap = params['overlap']
  if 'padding_mode' in params:
    padding_mode = params['eval_padding_mode']
  else:
    padding_mode = 'constant'
  if 'padding_val' in params:
    padding_val = params['padding_val']
  else:
    padding_val = 0

  roi_shape = np.array(params['val_input_shape_without_channel'])
  roi_shape_with_channel = np.append(roi_shape, NUM_CHANNELS)
  # If roi_shape=64,64,64 and overlap = 0.5
  # and image_shape is 121, 101, 111, 1
  strides = (roi_shape * (1 - overlap)).astype(np.int64)
  strides = np.append(strides, 1)  # Add channels to end.
  # strides = 32, 32, 32, 1
  strides = tf.constant(strides, dtype=tf.int64)

  # 25, 5, 15, 0 = 121, 101, 111, 1 % 32, 32, 32, 1
  leftover = image_shape % strides
  # 0, 1, 1, 0 = 25, 5, 15, 0 < 16, 16, 16, 0
  should_discard = tf.cast(leftover < strides // 2, tf.int64)
  # 0, 5, 15, 0 = 25, 5, 15, 0 * 0, 1, 1, 0
  discard_region = leftover * should_discard
  # 0, 2, 7, 0 = 0, 5, 15, 0 // 2
  left_discard = discard_region // 2
  # 121, 96, 96, 1 = 121, 101, 111, 1 - 0, 5, 15, 0
  image_shape_after_discard = image_shape - discard_region

  image = tf.slice(image, left_discard, image_shape_after_discard)
  label = tf.slice(label, left_discard, image_shape_after_discard)
  # 7, 32, 32, 1  = 32, 32, 32, 1 - 25, 5, 15, 0 * 1, 0, 0, 1
  pad_region = strides - (leftover * (1 - should_discard))
  # 7, 0, 0, 0 = 7, 32, 32, 1 % 32, 32, 32, 1
  pad_region = pad_region % strides
  # 128, 96, 96, 1 = 121, 96, 96, 1 + 7, 0, 0, 0
  # 0, 0, 0, 0 = 128, 96, 96, 1  < 64, 64, 64, 1
  requires_more_padding = ((image_shape_after_discard + pad_region) <
                           roi_shape_with_channel)
  requires_more_padding = tf.cast(requires_more_padding, tf.int64)

  # 7, 0, 0, 0
  pad_region = pad_region + requires_more_padding * strides
  # 3, 0, 0, 0 = 7, 0, 0, 0 // 2
  left_padding = pad_region // 2
  # 4, 0, 0, 0 = 7, 0, 0, 0 - 3, 0, 0, 0
  right_padding = pad_region - left_padding

  padding_conf = tf.transpose(tf.stack([left_padding, right_padding]))
  image = tf.pad(image, padding_conf, mode=padding_mode,
                 constant_values=padding_val)
  label = tf.pad(label, padding_conf, mode='constant', constant_values=0)
  # 128, 96,96, 1
  image_roi = image_shape_after_discard + pad_region
  stride_sizes = (image_roi - roi_shape_with_channel) // strides + 1

  make_sliding_windows_in_dataset = params.get(
      'make_sliding_windows_in_dataset', False)
  if not make_sliding_windows_in_dataset:
    # In this case, each batch is a single original image padded to the max
    # shape. We compute the sliding window slice indices in the dataset, and
    # append this information to the batch.
    padding_size = tf.constant(MAX_EVAL_IMAGE_SHAPE) - image_roi
    padding_size = tf.stack(
        [tf.constant([0, 0, 0, 0], dtype=tf.int64), padding_size])
    pad_to_max_eval_shape = tf.transpose(padding_size)
    image = tf.pad(image, pad_to_max_eval_shape)
    label = tf.pad(label, pad_to_max_eval_shape)
    # 64, 32, 32, 0 = 128, 96,96, 1 - 64, 64, 64, 1
    # 3, 2, 2, 1 = 64, 32, 32, 0 % 32, 32, 32, 0 + 1
    return {'image': image,
            'label': label,
            'image_roi': image_roi,
            'stride_sizes': stride_sizes,
            'left_padding': left_padding,
            'right_padding': right_padding,
            'is_padding': tf.constant(False),
            'image_idx': image_idx}
  else:
    # In this case, each batch is a sliding window, small puzzle peace.
    # image_indices and crop_locs tell us the location of the sliding window
    # in terms of which image it belongs to, and where is the crop location.
    (images, labels, crop_locs, image_indices, crop_masks, is_padding,
     norm_maps) = make_sliding_windows(image, label, image_roi, left_padding,
                                       right_padding, stride_sizes, image_idx,
                                       params)
    return {
        'image': images,
        'label': labels,
        'crop_locations': crop_locs,
        'image_idx': image_indices,
        'crop_masks': crop_masks,
        'is_padding': is_padding,
        'norm_maps': norm_maps,
    }


class RandBalancedCrop:
  """RandBalancedCrop augumentation."""

  def __init__(self, patch_size, oversampling):
    self.patch_size = patch_size
    self.patch_shape = (-1, patch_size[0], patch_size[1], patch_size[2])
    self.oversampling = oversampling

  def __call__(self, image, label, label1_oversample, label2_oversample):

    # pylint: disable=g-long-lambda
    image, label = tf.cond(
        tf.random.uniform(()) < self.oversampling,
        lambda: self.rand_foreg_cropd_tf(image, label,
                                         label1_oversample,
                                         label2_oversample),
        lambda: self._rand_crop_tf(image, label))
    # pylint: enable=g-long-lambda
    return image, label

  @staticmethod
  def randrange(max_range):
    return 0 if max_range == 0 else random.randrange(max_range)

  @staticmethod
  def randrange_tf(max_range):
    return tf.cond(
        tf.equal(max_range, 0), lambda: tf.constant(0),
        lambda: tf.random.uniform((), 0, max_range, dtype=tf.int32))

  def get_cords(self, cord, idx):
    return cord[idx], cord[idx] + self.patch_size[idx]

  def _rand_crop(self, image, label):
    """random crop."""
    ranges = [s - p for s, p in zip(image.shape[1:], self.patch_size)]
    cord = [self.randrange(x) for x in ranges]
    low_x, high_x = self.get_cords(cord, 0)
    low_y, high_y = self.get_cords(cord, 1)
    low_z, high_z = self.get_cords(cord, 2)
    image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
    label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
    return image, label

  def _rand_crop_tf(self, image, label):
    """random crop."""
    ranges = tf.shape(image)[1:] - self.patch_size
    cord = tf.map_fn(self.randrange_tf, ranges)
    low_x, high_x = self.get_cords(cord, 0)
    low_y, high_y = self.get_cords(cord, 1)
    low_z, high_z = self.get_cords(cord, 2)
    image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
    label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
    return image, label

  @staticmethod
  def scipy_wrapper(label_tensor, current_label):
    """Calls scipy implementation of find objects."""
    foreg_slices = scipy.ndimage.find_objects(
        scipy.ndimage.measurements.label(label_tensor == current_label)[0])
    foreg_slices = [x for x in foreg_slices if x is not None]
    slice_volumes = [
        np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices
    ]

    slice_idx = np.argsort(slice_volumes)[-2:]

    foreg_slices = [foreg_slices[i] for i in slice_idx]

    if not foreg_slices:
      found = False
      b0, e0, b1, e1 = [-1] * 4, [-1] * 4, [-1] * 4, [-1] * 4
    elif len(foreg_slices) == 1:
      found = True
      b0 = [fs.start for fs in foreg_slices[0]]
      e0 = [fs.stop for fs in foreg_slices[0]]
      b1 = b0
      e1 = e0
    else:
      found = True
      b0 = [fs.start for fs in foreg_slices[0]]
      e0 = [fs.stop for fs in foreg_slices[0]]
      b1 = [fs.start for fs in foreg_slices[1]]
      e1 = [fs.stop for fs in foreg_slices[1]]
    return found, b0, e0, b1, e1

  def rand_foreg_cropd_tf(self, image, label,
                          label1_oversample,
                          label2_oversample):
    """rand_foreg_cropd."""

    cl = tf.random.uniform((), 0, 2, dtype=tf.int32) + 1
    box_idx = tf.random.uniform((), 0, 2, dtype=tf.int32)

    oversample_params = tf.cond(
        tf.equal(cl, tf.constant(1, dtype=tf.int32)),
        lambda: label1_oversample, lambda: label2_oversample)
    found, b0, e0, b1, e1 = oversample_params

    def adjust_dimenson_tf(patch_size, b, e, idx, label_shape):
      diff = tf.cast(patch_size[idx - 1] - (e[idx] - b[idx]), tf.int32)
      sign = tf.cast(tf.cond(diff < 0, lambda: -1, lambda: 1), tf.int32)
      diff = tf.math.abs(diff)
      ladj = tf.random.uniform((), 0, diff, dtype=tf.int32)
      hadj = diff - ladj
      low = tf.math.maximum(0, b[idx] - sign * ladj)
      high = tf.math.minimum(label_shape[idx], e[idx] + sign * hadj)
      diff = patch_size[idx - 1] - (high - low)
      decrease_low = tf.cast(tf.logical_and(diff > 0, low != 0), tf.int32)
      increase_high = tf.cast(tf.logical_and(diff > 0, low == 0), tf.int32)
      high = high + diff * increase_high
      low = low - diff * decrease_low
      return low, high

    def oversample_crop(image, label, box_idx, b0, e0, b1, e1):
      b, e = tf.cond(tf.equal(box_idx, 0), lambda: (b0, e0), lambda: (b1, e1))
      label_shape = tf.shape(label)
      low_x, high_x = adjust_dimenson_tf(self.patch_size, b, e, 1, label_shape)
      low_y, high_y = adjust_dimenson_tf(self.patch_size, b, e, 2, label_shape)
      low_z, high_z = adjust_dimenson_tf(self.patch_size, b, e, 3, label_shape)
      image = tf.slice(image, [0, low_x, low_y, low_z],
                       [1, high_x - low_x, high_y - low_y, high_z - low_z])
      label = tf.slice(label, [0, low_x, low_y, low_z],
                       [1, high_x - low_x, high_y - low_y, high_z - low_z])
      return image, label

    image, label = tf.cond(
        found,
        lambda: oversample_crop(image, label, box_idx, b0, e0, b1, e1),
        lambda: self._rand_crop_tf(image, label))

    image = tf.reshape(image, self.patch_shape)
    label = tf.reshape(label, self.patch_shape)
    return image, label


class RandFlip:
  """Random flip for the image."""

  def __init__(self):
    self.axis = [2, 3, 4]
    self.prob = 1 / len(self.axis)

  def flip(self, image, label, axis):
    image = tf.reverse(image, axis=[axis])
    label = tf.reverse(label, axis=[axis])
    return image, label

  def __call__(self, data):
    for axis in self.axis:
      flip_fn = functools.partial(self.flip, data['image'], data['label'],
                                  axis)
      predicate = tf.random.uniform(shape=[], minval=0, maxval=1.0) < self.prob
      data['image'], data['label'] = tf.cond(
          predicate,
          lambda: flip_fn(),  # pylint: disable=unnecessary-lambda, cell-var-from-loop
          lambda: (data['image'], data['label']))
    return data


class Cast:
  """Cast dat type."""

  def __init__(self, types):
    self.types = types

  def __call__(self, data):
    data['image'] = tf.cast(data['image'], self.types[0])
    data['label'] = tf.cast(data['label'], self.types[1])
    return data


class RandomBrightnessAugmentation:
  """Random brightness agumentation."""

  def __init__(self, factor, prob):
    self.prob = prob
    self.factor = factor

  def __call__(self, data):
    image = data['image']
    predicate = tf.random.uniform(shape=[], minval=0, maxval=1.0) < self.prob
    image_scale = 1 + tf.random.uniform(shape=[],
                                        minval=1.0 - self.factor,
                                        maxval=1.0 + self.factor)
    image_scale = tf.cast(image_scale, image.dtype),
    image = tf.cond(
        predicate,
        lambda: tf.cast(image * image_scale, image.dtype),
        lambda: image)
    data.update({'image': image})
    return data


class GaussianNoise:
  """Add gaussian noise to the image."""

  def __init__(self, mean, std, prob):
    self.mean = mean
    self.std = std
    self.prob = prob

  def __call__(self, image):
    predicate = tf.random.uniform(shape=[],
                                  minval=0.0,
                                  maxval=1.0) < self.prob
    scale = tf.random.uniform(shape=[], minval=0.0, maxval=self.std,
                              dtype=image.dtype)
    noise = tf.random.normal(shape=image.shape,
                             mean=self.mean,
                             stddev=scale,
                             dtype=image.dtype)
    image = tf.cond(
        predicate,
        lambda: image + noise,
        lambda: image)
    return image


def make_eval_padding_dataset(params):
  """Create a fake padding dataset."""
  if params.get('use_bfloat16', False):
    dtype = tf.bfloat16
  else:
    dtype = tf.float32

  image_shape = params['val_input_shape']
  label_shape = params['val_input_shape']
  inp = np.zeros(image_shape).astype(np.float32)
  label = np.zeros(label_shape).astype(np.float32)

  make_sliding_windows_in_dataset = params.get(
      'make_sliding_windows_in_dataset', True)
  def transform_padding_data(data):
    image = data['image']
    label = data['label']
    image_shape = data['image_shape']
    image_idx = tf.constant(0, dtype=tf.int64)
    ds = prepare_sliding_window(image, label, image_idx, image_shape, params)
    if make_sliding_windows_in_dataset:
      ds['is_padding'] = tf.constant([True])
    else:
      ds['is_padding'] = tf.constant(True)
    return ds

  padding_dataset = tf.data.Dataset.from_tensors({
      'image': tf.cast(tf.constant(inp), dtype),
      'label': tf.constant(label),
      'image_shape': tf.constant(image_shape, dtype=tf.int64)
  }).map(transform_padding_data)
  if make_sliding_windows_in_dataset:
    padding_dataset = padding_dataset.unbatch()
  return padding_dataset


def scipy_wrapper_cache(label_tensor, current_label):
  """Calls scipy implementation of find objects."""
  foreg_slices = scipy.ndimage.find_objects(
      scipy.ndimage.measurements.label(label_tensor == current_label)[0])
  foreg_slices = [x for x in foreg_slices if x is not None]
  slice_volumes = [
      np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices
  ]

  slice_idx = np.argsort(slice_volumes)[-2:]

  foreg_slices = [foreg_slices[i] for i in slice_idx]

  if not foreg_slices:
    found = False
    b0, e0, b1, e1 = [-1] * 4, [-1] * 4, [-1] * 4, [-1] * 4
  elif len(foreg_slices) == 1:
    found = True
    b0 = [fs.start for fs in foreg_slices[0]]
    e0 = [fs.stop for fs in foreg_slices[0]]
    b1 = b0
    e1 = e0
  else:
    found = True
    b0 = [fs.start for fs in foreg_slices[0]]
    e0 = [fs.stop for fs in foreg_slices[0]]
    b1 = [fs.start for fs in foreg_slices[1]]
    e1 = [fs.stop for fs in foreg_slices[1]]
  return found, b0, e0, b1, e1


class InputFn(object):
  """Input function."""

  def __init__(self, file_list, params):
    self._file_list = file_list
    self._is_training = params['is_training']
    self._dataset_fn = functools.partial(
        tf.data.TFRecordDataset, compression_type='GZIP')
    self._parser_fn = self.create_parser_fn(params)
    self._postprocess_fn = self._create_postprocess_fn(params)
    self._rand_flip = RandFlip()
    image_dtype = (tf.bfloat16
                   if params.get('use_bfloat16', False) else tf.float32)
    self._cast = Cast(types=(image_dtype, tf.int32))
    self._rand_scale = RandomBrightnessAugmentation(factor=0.3, prob=0.1)
    self._gaussian_noise = GaussianNoise(mean=0.0, std=0.1, prob=0.1)
    self._rand_crop = RandBalancedCrop(
        patch_size=params['input_shape_without_channel'],
        oversampling=params['oversampling'])

  def create_parser_fn(self, params):
    """Create parse fn to extract tensors from tf.Example."""

    del params
    def _parser(serialized_example, image_idx):
      """Parses a single tf.Example into image and label tensors."""
      features = tf.io.parse_example(
          [serialized_example],
          features={
              'image': tf.io.FixedLenFeature([], dtype=tf.string),
              'image_shape': tf.io.FixedLenFeature([4], dtype=tf.int64),
              'label': tf.io.FixedLenFeature([], dtype=tf.string),
              'label_shape': tf.io.FixedLenFeature([4], dtype=tf.int64),
          })
      image = tf.io.decode_raw(features['image'], tf.float32)
      if isinstance(image, tf.SparseTensor):
        image = tf.sparse_tensor_to_dense(image)
      label = tf.io.decode_raw(features['label'], tf.float32)
      if isinstance(label, tf.SparseTensor):
        label = tf.sparse_tensor_to_dense(label)

      image_shape = tf.squeeze(features['image_shape'], axis=0)
      label_shape = tf.squeeze(features['label_shape'], axis=0)
      label_and_image_shape_check = tf.assert_equal(image_shape, label_shape)
      with tf.control_dependencies([label_and_image_shape_check]):
        image = tf.reshape(image, image_shape)
        label = tf.reshape(label, label_shape)

      data = {'image': image, 'label': label, 'image_shape': image_shape,
              'image_idx': image_idx}
      return data
    return _parser

  def _create_postprocess_fn(self, params):
    """Creates processing fns."""
    def _postprocess(ds_batch):
      image = ds_batch['image']
      label = ds_batch['label']
      image_shape = ds_batch['image_shape']
      image_idx = ds_batch['image_idx']
      if self._is_training:
        label = tf.cast(label, tf.int32)
        image, label = self._rand_crop(image, label,
                                       ds_batch['label1_oversample'],
                                       ds_batch['label2_oversample'])
        # In training cast image type to bf16 or f32.
        if params.get('use_bfloat16', False):
          image = tf.cast(image, tf.bfloat16)
        image, label, _ = _move_image_channels_to_last_dim(
            image, label,
            tf.constant([NUM_CHANNELS, *params['input_shape_without_channel']],
                        dtype=tf.int64))
        data = {'image': image, 'label': label, 'image_shape': image_shape,
                'is_padding': tf.constant(False)}
        return data

      else:
        # The images are stored so that they have [C, H, W, D].
        # Since C appears to be always 1, we simply reshape it without transpose
        # to make it [H, W, D, 1]
        image, label, image_shape = _move_image_channels_to_last_dim(
            image, label, image_shape)

        test_dataset_shape = params.get('test_dataset_shape', None)
        if test_dataset_shape:
          image, label, image_shape = _slice_image_for_testing(image, label,
                                                               image_shape,
                                                               params)
        return prepare_sliding_window(image, label, image_idx, image_shape,
                                      params)
    return _postprocess

  @tf.function
  def __call__(self, params):
    """Generates features and labels for training or evaluation.

    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.

    Args:
      params: model parameters in ParamsDict like object.

    Returns:
      tf.data.Dataset
    """
    make_sliding_windows_in_dataset = params.get(
        'make_sliding_windows_in_dataset', True)
    num_hosts = params.get('num_hosts', 1)
    host_index = params.get('host_index', 0)
    if self._is_training:
      if 'training_num_hosts' in params:
        tf.logging.info('Overwriting training_num_hosts old:%s now:%s',
                        num_hosts, params['training_num_hosts'])
        num_hosts = params['training_num_hosts']
      if 'training_host_index' in params:
        tf.logging.info('Overwriting training_host_index old:%s now:%s',
                        host_index, params['training_host_index'])
        host_index = params['training_host_index']
    if host_index >= num_hosts:
      raise ValueError('Dataset error: host_index:%d and num_hosts:%d' % (
          host_index, num_hosts))
    if self._is_training:
      batch_size = params['host_batch_size']
    else:
      batch_size = params['host_eval_batch_size']
    dataset = tf.data.Dataset.from_tensor_slices(self._file_list)
    # Assign a index to each image file.
    # Dataset consists of (image_idx, filename)
    dataset = dataset.enumerate()

    if self._is_training:

      # 168 samples first shuffled within themselves, and repeated.
      dataset = dataset.shuffle(
          params['shuffle_buffer_size'], seed=params['seed'],
          reshuffle_each_iteration=False)
      num_images = params['num_train_images']
      # How many repeats of the dataset we need for equal division to hosts.
      # 168 images, 32 hosts = 4 repeats
      # 168 * 4 can be evenly divided to 32 hosts.
      num_repeats = num_hosts // math.gcd(num_hosts, num_images)
      dataset = dataset.repeat(num_repeats).shard(num_hosts, host_index)
      # Dataset should have only 21 images if num_hosts = 32

    def read_dataset_and_append_image_index(image_idx, file_name):
      image_dataset = self._dataset_fn(file_name)
      image_dataset = image_dataset.map(lambda image: (image, image_idx))
      return image_dataset

    # After below, dataset consifsts of (image, image_idx)
    dataset = dataset.interleave(
        read_dataset_and_append_image_index,
        cycle_length=params['interleave_cycle_length'],
        num_parallel_calls=params['interleave_cycle_length'])

    dataset = dataset.map(self._parser_fn, num_parallel_calls=64)
    if self._is_training:
      def compute_and_cache_oversampling(data):
        label = data['label']
        label = tf.cast(label, tf.int32)
        found, b0, e0, b1, e1 = tf.py_function(
            func=scipy_wrapper_cache, inp=[label, 1],
            Tout=[tf.bool, tf.int32, tf.int32, tf.int32, tf.int32])
        found = tf.reshape(found, ())
        b0 = tf.reshape(b0, [4])
        e0 = tf.reshape(e0, [4])
        b1 = tf.reshape(b1, [4])
        e1 = tf.reshape(e1, [4])
        data['label1_oversample'] = (found, b0, e0, b1, e1)
        found, b0, e0, b1, e1 = tf.py_function(
            func=scipy_wrapper_cache, inp=[label, 2],
            Tout=[tf.bool, tf.int32, tf.int32, tf.int32, tf.int32])
        found = tf.reshape(found, ())
        b0 = tf.reshape(b0, [4])
        e0 = tf.reshape(e0, [4])
        b1 = tf.reshape(b1, [4])
        e1 = tf.reshape(e1, [4])
        data['label2_oversample'] = (found, b0, e0, b1, e1)
        return data
      dataset = dataset.map(compute_and_cache_oversampling,
                            num_parallel_calls=64, deterministic=True)
      dataset = dataset.cache().repeat()

    else:
      # Shard the eval dataset here, only if
      # make_sliding_windows_in_dataset=false
      if not make_sliding_windows_in_dataset:
        dataset = dataset.shard(num_hosts, host_index)

    # Parses the fetched records to input tensors for model function.
    # Training dataset will have a dictionary
    #   {'image','label', 'image_shape', 'is_padding'}
    # Eval dataset
    # 1- make_sliding_windows_in_dataset = True
    #   * 'image': This is going to be in eval_input shape:
    #       [num_windows, 128, 128, 128, 1]
    #   * 'label':  This is going to be in eval_input shape:
    #       [num_windows, 128, 128, 128, 1]
    #   * 'crop_locations': [num_windows, 4], The beginning slice loc of
    #       of each window. E.g. [[0,0,0,0], [0,64,64,0] ]
    #   * 'crop_masks': [num_windows, 128, 128, 128, 1], this is a 0,1 mask,
    #     how to mask the result of sliding window. (it is 1 if it corresponds
    #     to real image, 0 if image padding)
    #   * 'is_padding': Whether this is a real input or not. bool in shape of
    #       [num_windows]
    #   * 'image_idx': The image index with shape [num_windows], between [0, 41]
    #   * 'norm_maps': with shape [num_windows, 128, 128, 128, 1], value to
    #      divide the image result.
    # 2- make_sliding_windows_in_dataset = False
    #   * 'image': in the shape padded to make shape. 448, 448, 448, 1
    #   * 'label':  in the shape padded to make shape. 448, 448, 448, 1
    #   * 'image_roi':The actual size of image (this is after padding ops)
    #   * 'stride_sizes': How many strides we need per dimension.
    #       [h_stride, w_stride, d_stride, 1]
    #   * 'left_padding': how much padding is applied from left side of the
    #       image. [h_lp, w_lp, d_lp, 1]. This needs to be discarded from
    #       results when computing scores.
    #   * 'right_padding': how much padding is applied from right side of the
    #       image. [h_rp, w_rp, d_rp, 1]. This needs to be discarded from
    #       results when computing scores.
    #   * 'is_padding': Whether this is a real input or not. [False]
    #   * 'image_idx': The image index, between [0, 41].
    # TODO: experiment with parallel calls here.
    dataset = dataset.map(self._postprocess_fn, num_parallel_calls=64)

    if not self._is_training:

      if make_sliding_windows_in_dataset:
        # Unbatch so that each sliding window is a seperate batch.
        dataset = dataset.unbatch()
        # Shard the dataset here for better load imbalance accross hosts.
        dataset = dataset.shard(num_hosts, host_index)
        per_host_eval_samples = int(params['num_eval_steps'] * batch_size)
      else:
        per_host_eval_samples = int(params['num_eval_steps'] * batch_size)

      padding_dataset = make_eval_padding_dataset(params).repeat(
          per_host_eval_samples)
      dataset = dataset.concatenate(padding_dataset).take(per_host_eval_samples)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _transform_train_dataset(data):
      data = self._rand_flip(data)
      data = self._cast(data)
      data = self._rand_scale(data)
      data['image'] = self._gaussian_noise(data['image'])
      if params.get('use_bfloat16', False):
        assert data['image'].dtype == tf.bfloat16
      return data

    if self._is_training:
      dataset = dataset.map(_transform_train_dataset,
                            num_parallel_calls=64)
    else:
      dataset = dataset.cache()
      dataset = dataset.repeat()
    dataset = dataset.prefetch(1)
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 64
    dataset = dataset.with_options(options)
    return dataset


class FakeDatasetFn(object):
  """Input function."""

  def __init__(self, params):
    self._is_training = params['is_training']

  def __call__(self, params):
    """Generates features and labels for training or evaluation.

    This uses the input pipeline based approach using file name queue
    to read data so that entire data is not loaded in memory.

    Args:
      params: model parameters in ParamsDict like object.

    Returns:
      tf.data.Dataset
    """
    make_sliding_windows_in_dataset = params.get(
        'make_sliding_windows_in_dataset', True)
    if params.get('use_bfloat16', False):
      dtype = tf.bfloat16
    else:
      dtype = tf.float32

    if self._is_training:
      batch_size = params['host_batch_size']
      image_shape = params['input_shape']
      label_shape = params['input_shape']
    else:
      batch_size = params['host_eval_batch_size']
      image_shape = [200, 256, 315, 1]
      label_shape = image_shape
    create_nan_data = params.get('fake_nan_data', False)

    inp = np.random.randn(*image_shape).astype(np.float32)
    label = np.random.randint(2, size=label_shape).astype(np.int32)
    if create_nan_data:
      inp += np.nan
    dataset = tf.data.Dataset.from_tensors({
        'image': tf.cast(tf.constant(inp), dtype),
        'label': tf.constant(label),
        'image_shape': tf.constant(image_shape, dtype=tf.int64),
        'is_padding': tf.constant(True)})
    if not self._is_training:
      # pylint: disable=g-long-lambda
      dataset = dataset.map(lambda x: prepare_sliding_window(
          x['image'], x['label'], tf.constant(0, dtype=tf.int64),
          x['image_shape'], params))
      # pylint: enable=g-long-lambda

      if make_sliding_windows_in_dataset:
        # Unbatch so that each sliding window is a seperate batch.
        dataset = dataset.unbatch()

    dataset = dataset.repeat()
    if not self._is_training:
      def make_random_image_idx(dataset_element):
        new_image_idx = tf.random.uniform(shape=[], minval=0,
                                          maxval=params['num_eval_images'],
                                          dtype=tf.dtypes.int32)

        dataset_element['image_idx'] = new_image_idx
        return dataset_element
      dataset = dataset.map(make_random_image_idx)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

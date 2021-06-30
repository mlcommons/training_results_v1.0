"""Scripts to convert *x.npy and *y.npy into one single tfrecord file."""

import os
from typing import Sequence

from absl import app
import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED import gfile


def _bytes_feature(value):
  """Returns a float_list from numpy array."""
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[value.reshape([-1]).tobytes()]))


def _int64_feature(value):
  """Returns an int64_list from a tulpe of int."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(image, label):
  """Creates a tf.train.Example message ready to be written to a file."""
  feature = {
      "image": _bytes_feature(image.astype(np.float32)),
      "image_shape": _int64_feature(np.array(image.shape)),
      "label": _bytes_feature(label.astype(np.float32)),
      "label_shape": _int64_feature(np.array(label.shape)),
  }
  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  path = "/REDACTED/nm-d/home/tpu-perf-team/unet3d/kits19/numpy_data"
  image_files = sorted(tf.io.gfile.glob(os.path.join(path, "*_x.npy")))
  for image_file in image_files:
    with tf.io.gfile.GFile(image_file, "rb") as f:
      image = np.load(f)
    label_file = image_file.replace("_x", "_y")
    with tf.io.gfile.GFile(label_file, "rb") as f:
      label = np.load(f)
    filename = image_file.replace("_x.npy", ".tfrecord")

    with gfile.Open(filename, "wb") as f:
      options = tf.python_io.TFRecordOptions(
          tf.python_io.TFRecordCompressionType.GZIP)
      with tf.python_io.TFRecordWriter(filename, options=options) as w:
        w.write(serialize_example(image, label))
      print("saved: ", filename)


if __name__ == "__main__":
  app.run(main)

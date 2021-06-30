# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from absl import flags
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4,
                     num_eval_samples=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    if "dataset_num_shards" in params and "dataset_index" in params:
      num_hosts = params["dataset_num_shards"]
      host_index = params["dataset_index"]
    else:
      num_hosts = 1
      host_index = 0

    num_dataset_per_shard = max(
        1,
        int(
            math.ceil(num_eval_samples / FLAGS.eval_batch_size) *
            FLAGS.eval_batch_size / num_hosts))
    tf.logging.info("num_dataset_per_shard: %d" % num_dataset_per_shard)

    def _float_feature(values):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))

    def _int64_feature(values):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    padded_dataset = tf.data.Dataset.from_tensors(
        tf.constant(
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "input_ids":
                            _int64_feature([0] * max_seq_length),
                        "input_mask":
                            _int64_feature([0] * max_seq_length),
                        "segment_ids":
                            _int64_feature([0] * max_seq_length),
                        "masked_lm_positions":
                            _int64_feature([0] * max_predictions_per_seq),
                        "masked_lm_ids":
                            _int64_feature([0] * max_predictions_per_seq),
                        "masked_lm_weights":
                            _float_feature([0] * max_predictions_per_seq),
                        "next_sentence_labels":
                            _int64_feature([0]),
                    })).SerializeToString(),
            dtype=tf.string)).repeat(num_dataset_per_shard)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      if not input_files:
        tf.logging.info("Using dummy data for training.")
        d = padded_dataset.repeat(batch_size * 100)
        input_length = 1
      else:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        input_length = len(input_files)

      if num_hosts > 1:
        input_length = int(math.ceil(input_length / num_hosts))
        tf.logging.info(
            "Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d"
            % (host_index, num_hosts))

        d = d.shard(num_hosts, host_index)
      d = d.shuffle(buffer_size=input_length)

      # `cycle_length` is the number of parallel files that get read and the
      # constant is based on empirical evidence.
      cycle_length = 10

      if input_files:
        # `determenistic=False` mode means that the interleaving is not exact.
        # This helps to avoid head-of-the-line blocking.
        d = d.interleave(
            tf.data.TFRecordDataset,
            deterministic=not is_training,
            cycle_length=cycle_length,
            num_parallel_calls=tf.data.AUTOTUNE)
      d = d.shuffle(buffer_size=1000)
      d = d.repeat()
    else:
      if not input_files:
        tf.logging.info("Using dummy data for eval.")
        d = padded_dataset.repeat(num_eval_samples)
      else:
        d = tf.data.TFRecordDataset(input_files)
      tf.logging.info(
          "Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d" %
          (host_index, num_hosts))
      d = d.take(num_eval_samples)
      d = d.shard(num_hosts, host_index)

      if num_eval_samples and num_eval_samples > 0:
        d = d.concatenate(padded_dataset).take(num_dataset_per_shard)
        tf.logging.info(
            "Padding the dataset: input_pipeline_id=%d padded_size=%d" %
            (host_index, num_dataset_per_shard - num_eval_samples / num_hosts))

      d = d.repeat()

    if not is_training or FLAGS.batch_size_buckets is None or len(
        FLAGS.batch_size_buckets) == 1:
      d = d.map(
          lambda record: _decode_record(record, name_to_features),
          num_parallel_calls=num_cpu_threads)
      d = d.batch(batch_size=batch_size, drop_remainder=True)
      return d

    d = d.map(lambda record: _decode_record(record, name_to_features),
              num_cpu_threads)
    seq_len_to_bucket = tf.cumsum(
        [1 if i in FLAGS.seq_len_buckets else 0 for i in range(max_seq_length)])

    def index_fn(record):
      return tf.cast(
          tf.gather(seq_len_to_bucket,
                    tf.reduce_sum(record["input_mask"]) - 1), tf.int64)

    def window_fn(k):
      return tf.cast(tf.gather(FLAGS.batch_size_buckets, k), tf.int64)

    def batch_fn(bucket, dataset):

      def pad_to_max_batch(tensor):
        return tf.slice(
            tf.pad(tensor, [[0, max(FLAGS.batch_size_buckets)], [0, 0]]),
            [0, 0], [max(FLAGS.batch_size_buckets), -1])

      return dataset.batch(
          tf.cast(tf.gather(FLAGS.batch_size_buckets, bucket), tf.int64)).map(
              lambda t: {k: pad_to_max_batch(t[k]) for k in t}, num_cpu_threads)

    d = d.group_by_window(index_fn, batch_fn, window_size_func=window_fn)
    return d

  return input_fn

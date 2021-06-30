# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import os
import tqdm
import time
import argparse
import glob
import struct
import numpy as np
from tensorflow.compat import v1 as tf
from functools import reduce
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
tf.enable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument("--tf-record-glob", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--max-sequence-length", help="The maximum number of tokens in an example", default=512, type=int)
parser.add_argument("--max-predictions-per-sequence", help="The maximum number of masked tokens in an un-packed example", default=76, type=int)
args = parser.parse_args()

input_files = glob.glob(args.tf_record_glob)
assert len(input_files) > 0

# Original feature names
name_to_features = {
        "input_ids": tf.FixedLenFeature([args.max_sequence_length], tf.int64),
        "input_mask": tf.FixedLenFeature([args.max_sequence_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([args.max_sequence_length], tf.int64),
        "masked_lm_positions": tf.FixedLenFeature([args.max_predictions_per_sequence], tf.int64),
        "masked_lm_ids": tf.FixedLenFeature([args.max_predictions_per_sequence], tf.int64),
        "masked_lm_weights": tf.FixedLenFeature([args.max_predictions_per_sequence], tf.float32),
        "next_sentence_labels": tf.FixedLenFeature([1], tf.int64)
    }

# Convert the input files
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)


def convert_file(file):
    d = tf.data.TFRecordDataset(file)
    d = d.map(lambda record: tf.parse_single_example(record, name_to_features))

    output_file = os.path.join(args.output_path, os.path.basename(file))

    with open(output_file, "wb") as writer:
        for example in d:
            # Pack into binary format
            line = reduce(lambda accl, i: accl + struct.pack('<I', i),
                          chain(example["input_ids"].numpy().astype(np.int32),
                                example["input_mask"].numpy().astype(np.int32),
                                example["segment_ids"].numpy().astype(np.int32),
                                example["masked_lm_positions"].numpy().astype(np.int32),
                                example["masked_lm_ids"].numpy().astype(np.int32),
                                example["masked_lm_weights"].numpy().astype(np.int32),
                                example["next_sentence_labels"].numpy().astype(np.int32)), b'')
            writer.write(line)

start = time.time()
with ProcessPoolExecutor(16) as executor:
    for partial_result in executor.map(convert_file, input_files):
        pass
print(f"\nDone. Took: {time.time() - start:3.2f} seconds to convert dataset.")

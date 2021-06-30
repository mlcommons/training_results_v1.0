"""Support code to translate the PyTorch checkpoint to TF.

The input is the PyT checkpoint in pickle format
Directionary checkpoint_tensor_mappings
has a 1-1 mapping for tensors between the two frameworks
"""

import numpy as np
import tensorflow.compat.v1 as tf
from REDACTED.tensorflow.python.lib.io import file_io as _file_io

# I tried to find an easy to understand logical mapping from pyt to tf
# but I could not so I gave up and resorted to this brute force mapping
checkpoint_tensor_mappings = {
    "resnet34/conv2d/kernel:0":
        "conv1.weight",
    "resnet34/batch_normalization/beta:0":
        "bn1.bias",
    "resnet34/batch_normalization/gamma:0":
        "bn1.weight",
    "resnet34/batch_normalization/moving_mean:0":
        "bn1.running_mean",
    "resnet34/batch_normalization/moving_variance:0":
        "bn1.running_var",
    "resnet34/conv2d_1/kernel:0":
        "layer1.0.conv1.weight",
    "resnet34/batch_normalization_1/beta:0":
        "layer1.0.bn1.bias",
    "resnet34/batch_normalization_1/gamma:0":
        "layer1.0.bn1.weight",
    "resnet34/batch_normalization_1/moving_mean:0":
        "layer1.0.bn1.running_mean",
    "resnet34/batch_normalization_1/moving_variance:0":
        "layer1.0.bn1.running_var",
    "resnet34/conv2d_2/kernel:0":
        "layer1.0.conv2.weight",
    "resnet34/batch_normalization_2/beta:0":
        "layer1.0.bn2.bias",
    "resnet34/batch_normalization_2/gamma:0":
        "layer1.0.bn2.weight",
    "resnet34/batch_normalization_2/moving_mean:0":
        "layer1.0.bn2.running_mean",
    "resnet34/batch_normalization_2/moving_variance:0":
        "layer1.0.bn2.running_var",
    "resnet34/conv2d_3/kernel:0":
        "layer1.1.conv1.weight",
    "resnet34/batch_normalization_3/beta:0":
        "layer1.1.bn1.bias",
    "resnet34/batch_normalization_3/gamma:0":
        "layer1.1.bn1.weight",
    "resnet34/batch_normalization_3/moving_mean:0":
        "layer1.1.bn1.running_mean",
    "resnet34/batch_normalization_3/moving_variance:0":
        "layer1.1.bn1.running_var",
    "resnet34/conv2d_4/kernel:0":
        "layer1.1.conv2.weight",
    "resnet34/batch_normalization_4/beta:0":
        "layer1.1.bn2.bias",
    "resnet34/batch_normalization_4/gamma:0":
        "layer1.1.bn2.weight",
    "resnet34/batch_normalization_4/moving_mean:0":
        "layer1.1.bn2.running_mean",
    "resnet34/batch_normalization_4/moving_variance:0":
        "layer1.1.bn2.running_var",
    "resnet34/conv2d_5/kernel:0":
        "layer1.2.conv1.weight",
    "resnet34/batch_normalization_5/beta:0":
        "layer1.2.bn1.bias",
    "resnet34/batch_normalization_5/gamma:0":
        "layer1.2.bn1.weight",
    "resnet34/batch_normalization_5/moving_mean:0":
        "layer1.2.bn1.running_mean",
    "resnet34/batch_normalization_5/moving_variance:0":
        "layer1.2.bn1.running_var",
    "resnet34/conv2d_6/kernel:0":
        "layer1.2.conv2.weight",
    "resnet34/batch_normalization_6/beta:0":
        "layer1.2.bn2.bias",
    "resnet34/batch_normalization_6/gamma:0":
        "layer1.2.bn2.weight",
    "resnet34/batch_normalization_6/moving_mean:0":
        "layer1.2.bn2.running_mean",
    "resnet34/batch_normalization_6/moving_variance:0":
        "layer1.2.bn2.running_var",
    "resnet34/conv2d_7/kernel:0":
        "layer2.0.downsample.0.weight",
    "resnet34/batch_normalization_7/beta:0":
        "layer2.0.downsample.1.bias",
    "resnet34/batch_normalization_7/gamma:0":
        "layer2.0.downsample.1.weight",
    "resnet34/batch_normalization_7/moving_mean:0":
        "layer2.0.downsample.1.running_mean",
    "resnet34/batch_normalization_7/moving_variance:0":
        "layer2.0.downsample.1.running_var",
    "resnet34/conv2d_8/kernel:0":
        "layer2.0.conv1.weight",
    "resnet34/batch_normalization_8/beta:0":
        "layer2.0.bn1.bias",
    "resnet34/batch_normalization_8/gamma:0":
        "layer2.0.bn1.weight",
    "resnet34/batch_normalization_8/moving_mean:0":
        "layer2.0.bn1.running_mean",
    "resnet34/batch_normalization_8/moving_variance:0":
        "layer2.0.bn1.running_var",
    "resnet34/conv2d_9/kernel:0":
        "layer2.0.conv2.weight",
    "resnet34/batch_normalization_9/beta:0":
        "layer2.0.bn2.bias",
    "resnet34/batch_normalization_9/gamma:0":
        "layer2.0.bn2.weight",
    "resnet34/batch_normalization_9/moving_mean:0":
        "layer2.0.bn2.running_mean",
    "resnet34/batch_normalization_9/moving_variance:0":
        "layer2.0.bn2.running_var",
    "resnet34/conv2d_10/kernel:0":
        "layer2.1.conv1.weight",
    "resnet34/batch_normalization_10/beta:0":
        "layer2.1.bn1.bias",
    "resnet34/batch_normalization_10/gamma:0":
        "layer2.1.bn1.weight",
    "resnet34/batch_normalization_10/moving_mean:0":
        "layer2.1.bn1.running_mean",
    "resnet34/batch_normalization_10/moving_variance:0":
        "layer2.1.bn1.running_var",
    "resnet34/conv2d_11/kernel:0":
        "layer2.1.conv2.weight",
    "resnet34/batch_normalization_11/beta:0":
        "layer2.1.bn2.bias",
    "resnet34/batch_normalization_11/gamma:0":
        "layer2.1.bn2.weight",
    "resnet34/batch_normalization_11/moving_mean:0":
        "layer2.1.bn2.running_mean",
    "resnet34/batch_normalization_11/moving_variance:0":
        "layer2.1.bn2.running_var",
    "resnet34/conv2d_12/kernel:0":
        "layer2.2.conv1.weight",
    "resnet34/batch_normalization_12/beta:0":
        "layer2.2.bn1.bias",
    "resnet34/batch_normalization_12/gamma:0":
        "layer2.2.bn1.weight",
    "resnet34/batch_normalization_12/moving_mean:0":
        "layer2.2.bn1.running_mean",
    "resnet34/batch_normalization_12/moving_variance:0":
        "layer2.2.bn1.running_var",
    "resnet34/conv2d_13/kernel:0":
        "layer2.2.conv2.weight",
    "resnet34/batch_normalization_13/beta:0":
        "layer2.2.bn2.bias",
    "resnet34/batch_normalization_13/gamma:0":
        "layer2.2.bn2.weight",
    "resnet34/batch_normalization_13/moving_mean:0":
        "layer2.2.bn2.running_mean",
    "resnet34/batch_normalization_13/moving_variance:0":
        "layer2.2.bn2.running_var",
    "resnet34/conv2d_14/kernel:0":
        "layer2.3.conv1.weight",
    "resnet34/batch_normalization_14/beta:0":
        "layer2.3.bn1.bias",
    "resnet34/batch_normalization_14/gamma:0":
        "layer2.3.bn1.weight",
    "resnet34/batch_normalization_14/moving_mean:0":
        "layer2.3.bn1.running_mean",
    "resnet34/batch_normalization_14/moving_variance:0":
        "layer2.3.bn1.running_var",
    "resnet34/conv2d_15/kernel:0":
        "layer2.3.conv2.weight",
    "resnet34/batch_normalization_15/beta:0":
        "layer2.3.bn2.bias",
    "resnet34/batch_normalization_15/gamma:0":
        "layer2.3.bn2.weight",
    "resnet34/batch_normalization_15/moving_mean:0":
        "layer2.3.bn2.running_mean",
    "resnet34/batch_normalization_15/moving_variance:0":
        "layer2.3.bn2.running_var",
    "resnet34/conv2d_16/kernel:0":
        "layer3.0.downsample.0.weight",
    "resnet34/batch_normalization_16/beta:0":
        "layer3.0.bn2.bias",
    "resnet34/batch_normalization_16/gamma:0":
        "layer3.0.bn2.weight",
    "resnet34/batch_normalization_16/moving_mean:0":
        "layer3.0.downsample.1.running_mean",
    "resnet34/batch_normalization_16/moving_variance:0":
        "layer3.0.downsample.1.running_var",
    "resnet34/conv2d_17/kernel:0":
        "layer3.0.conv1.weight",
    "resnet34/batch_normalization_17/beta:0":
        "layer3.0.bn1.bias",
    "resnet34/batch_normalization_17/gamma:0":
        "layer3.0.bn1.weight",
    "resnet34/batch_normalization_17/moving_mean:0":
        "layer3.0.bn1.running_mean",
    "resnet34/batch_normalization_17/moving_variance:0":
        "layer3.0.bn1.running_var",
    "resnet34/conv2d_18/kernel:0":
        "layer3.0.conv2.weight",
    "resnet34/batch_normalization_18/beta:0":
        "layer3.0.downsample.1.bias",
    "resnet34/batch_normalization_18/gamma:0":
        "layer3.0.downsample.1.weight",
    "resnet34/batch_normalization_18/moving_mean:0":
        "layer3.0.bn2.running_mean",
    "resnet34/batch_normalization_18/moving_variance:0":
        "layer3.0.bn2.running_var",
    "resnet34/conv2d_19/kernel:0":
        "layer3.1.conv1.weight",
    "resnet34/batch_normalization_19/beta:0":
        "layer3.1.bn1.bias",
    "resnet34/batch_normalization_19/gamma:0":
        "layer3.1.bn1.weight",
    "resnet34/batch_normalization_19/moving_mean:0":
        "layer3.1.bn1.running_mean",
    "resnet34/batch_normalization_19/moving_variance:0":
        "layer3.1.bn1.running_var",
    "resnet34/conv2d_20/kernel:0":
        "layer3.1.conv2.weight",
    "resnet34/batch_normalization_20/beta:0":
        "layer3.1.bn2.bias",
    "resnet34/batch_normalization_20/gamma:0":
        "layer3.1.bn2.weight",
    "resnet34/batch_normalization_20/moving_mean:0":
        "layer3.1.bn2.running_mean",
    "resnet34/batch_normalization_20/moving_variance:0":
        "layer3.1.bn2.running_var",
    "resnet34/conv2d_21/kernel:0":
        "layer3.2.conv1.weight",
    "resnet34/batch_normalization_21/beta:0":
        "layer3.2.bn1.bias",
    "resnet34/batch_normalization_21/gamma:0":
        "layer3.2.bn1.weight",
    "resnet34/batch_normalization_21/moving_mean:0":
        "layer3.2.bn1.running_mean",
    "resnet34/batch_normalization_21/moving_variance:0":
        "layer3.2.bn1.running_var",
    "resnet34/conv2d_22/kernel:0":
        "layer3.2.conv2.weight",
    "resnet34/batch_normalization_22/beta:0":
        "layer3.2.bn2.bias",
    "resnet34/batch_normalization_22/gamma:0":
        "layer3.2.bn2.weight",
    "resnet34/batch_normalization_22/moving_mean:0":
        "layer3.2.bn2.running_mean",
    "resnet34/batch_normalization_22/moving_variance:0":
        "layer3.2.bn2.running_var",
    "resnet34/conv2d_23/kernel:0":
        "layer3.3.conv1.weight",
    "resnet34/batch_normalization_23/beta:0":
        "layer3.3.bn1.bias",
    "resnet34/batch_normalization_23/gamma:0":
        "layer3.3.bn1.weight",
    "resnet34/batch_normalization_23/moving_mean:0":
        "layer3.3.bn1.running_mean",
    "resnet34/batch_normalization_23/moving_variance:0":
        "layer3.3.bn1.running_var",
    "resnet34/conv2d_24/kernel:0":
        "layer3.3.conv2.weight",
    "resnet34/batch_normalization_24/beta:0":
        "layer3.3.bn2.bias",
    "resnet34/batch_normalization_24/gamma:0":
        "layer3.3.bn2.weight",
    "resnet34/batch_normalization_24/moving_mean:0":
        "layer3.3.bn2.running_mean",
    "resnet34/batch_normalization_24/moving_variance:0":
        "layer3.3.bn2.running_var",
    "resnet34/conv2d_25/kernel:0":
        "layer3.4.conv1.weight",
    "resnet34/batch_normalization_25/beta:0":
        "layer3.4.bn1.bias",
    "resnet34/batch_normalization_25/gamma:0":
        "layer3.4.bn1.weight",
    "resnet34/batch_normalization_25/moving_mean:0":
        "layer3.4.bn1.running_mean",
    "resnet34/batch_normalization_25/moving_variance:0":
        "layer3.4.bn1.running_var",
    "resnet34/conv2d_26/kernel:0":
        "layer3.4.conv2.weight",
    "resnet34/batch_normalization_26/beta:0":
        "layer3.4.bn2.bias",
    "resnet34/batch_normalization_26/gamma:0":
        "layer3.4.bn2.weight",
    "resnet34/batch_normalization_26/moving_mean:0":
        "layer3.4.bn2.running_mean",
    "resnet34/batch_normalization_26/moving_variance:0":
        "layer3.4.bn2.running_var",
    "resnet34/conv2d_27/kernel:0":
        "layer3.5.conv1.weight",
    "resnet34/batch_normalization_27/beta:0":
        "layer3.5.bn1.bias",
    "resnet34/batch_normalization_27/gamma:0":
        "layer3.5.bn1.weight",
    "resnet34/batch_normalization_27/moving_mean:0":
        "layer3.5.bn1.running_mean",
    "resnet34/batch_normalization_27/moving_variance:0":
        "layer3.5.bn1.running_var",
    "resnet34/conv2d_28/kernel:0":
        "layer3.5.conv2.weight",
    "resnet34/batch_normalization_28/beta:0":
        "layer3.5.bn2.bias",
    "resnet34/batch_normalization_28/gamma:0":
        "layer3.5.bn2.weight",
    "resnet34/batch_normalization_28/moving_mean:0":
        "layer3.5.bn2.running_mean",
    "resnet34/batch_normalization_28/moving_variance:0":
        "layer3.5.bn2.running_var"
}


def checkpoint_map(pyt_pickle_file, tf_checkpoint_filename):
  """Map pickle pyt tensors to tf tensors."""
  pickle_data = np.load(_file_io.FileIO(pyt_pickle_file, mode="rb"),
                        allow_pickle=True)

  _ = tf.compat.v1.train.get_or_create_global_step()
  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      None, None, None,
      job_name="tpu_worker")
  master = tpu_cluster_resolver.get_master()
  sess = tf.compat.v1.Session(target=master, graph=None, config=None)
  sess.run(tf.compat.v1.global_variables_initializer())

  # TF has momentum tensors that need to be setup.
  for v in tf.global_variables():
    if "beta" in v.name or "gamma" in v.name or "conv" in v.name:
      tmp_name = v.name[len("resnet34/"):len(v.name)-2]
      momentum_tensor_data = tf.zeros(shape=v.shape, dtype=tf.float32)
      if "beta" in v.name:
        momentum_tensor_name = tmp_name.replace("beta", "beta/Momentum")
      elif "gamma" in v.name:
        momentum_tensor_name = tmp_name.replace("gamma", "gamma/Momentum")
      elif "conv" in v.name:
        momentum_tensor_name = tmp_name.replace("kernel", "kernel/Momentum")
      tf.compat.v1.Variable(
          momentum_tensor_data, name=momentum_tensor_name, trainable=True)

  sess.run(tf.compat.v1.global_variables_initializer())

  for v in tf.global_variables():
    if v.name not in checkpoint_tensor_mappings.keys():
      print("Did not find tensor in TF checkpoint: ", v.name)
      continue
    pt_name = checkpoint_tensor_mappings[v.name]
    if pt_name not in pickle_data.keys():
      continue
    else:
      if "conv" in v.name:
        tensor_data = np.transpose(pickle_data[pt_name], (3, 2, 1, 0))
        v = tf.assign(v, tensor_data)
        sess.run(v)
      else:
        tensor_data = pickle_data[pt_name]
        v = tf.assign(v, tensor_data)
        sess.run(v)

  print("Saving checkpoint")
  saver = tf.train.Saver()
  saver.save(sess, tf_checkpoint_filename)
  assert False


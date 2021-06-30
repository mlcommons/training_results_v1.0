"""Training script for DLRM model."""
import functools
import queue
import threading

import REDACTED

from absl import app as absl_app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

from REDACTED.tensorflow.python.tpu import device_assignment
from REDACTED.tensorflow.python.tpu import tpu_embedding
from REDACTED.mlperf.submissions.training.v1_0.models.dlrm import common
from REDACTED.mlperf.submissions.training.v1_0.models.dlrm import dataloader
from REDACTED.mlperf.submissions.training.v1_0.models.dlrm import dlrm
from REDACTED.mlperf.submissions.training.v1_0.models.dlrm import dlrm_embedding_runner
from REDACTED.mlperf.submissions.training.v1_0.models.dlrm import feature_config as fc
from REDACTED.mlperf.submissions.training.v1_0.models.dlrm import utils
from REDACTED.mlperf.submissions.training.v1_0.models.mlp_log import mlp_log
from REDACTED.mlperf.submissions.training.v1_0.models.util import train_and_eval_runner as tr
from REDACTED.mlperf.submissions.training.v1_0.models.util.roc_metrics.python import roc_metrics

FLAGS = flags.FLAGS

_NUM_TRAIN_EXAMPLES = 4195197692
_NUM_EVAL_EXAMPLES = 89137318

_ACCURACY_THRESH = 0.8025


def get_input_fns(params, feature_config):
  """Returns input function objects."""

  def _csv_record_path(mode):
    return "{data_dir}/terabyte_processed_golden_shuffled/{mode}/{mode}*".format(
        data_dir=FLAGS.data_dir, mode=mode)

  def _batched_tfrecord_path(mode):
    replica_batch_size = params["batch_size"] // params["num_shards"]
    filecnt_path = (
        "{data_dir}/terabyte_tfrecords_batched{bs}/{mode}*/*".format(
            data_dir=FLAGS.data_dir,
            bs=replica_batch_size,
            mode=mode))
    file_cnt = len(tf.gfile.Glob(filecnt_path))

    return "{data_dir}/terabyte_tfrecords_batched{bs}/{mode}{file_cnt}shards/{mode}*".format(
        data_dir=FLAGS.data_dir,
        bs=replica_batch_size,
        mode=mode,
        file_cnt=file_cnt)

  num_datasets = FLAGS.num_tpu_shards // FLAGS.replicas_per_host

  if FLAGS.use_batched_tfrecords:
    train_input_fn = dataloader.CriteoTFRecordReader(
        file_path=_batched_tfrecord_path("train"),
        feature_config=feature_config,
        is_training=True,
        use_cached_data=params["use_cached_data"],
        use_synthetic_data=params["use_synthetic_data"],
        num_datasets=num_datasets,
        params=params)
    eval_input_fn = dataloader.CriteoTFRecordReader(
        file_path=_batched_tfrecord_path("eval"),
        feature_config=feature_config,
        is_training=False,
        use_cached_data=params["use_cached_data"],
        use_synthetic_data=params["use_synthetic_data"],
        num_datasets=num_datasets,
        params=params)
  else:
    train_input_fn = dataloader.CriteoTsvReader(
        file_path=_csv_record_path("train"),
        feature_config=feature_config,
        is_training=True,
        parallelism=16,
        use_cached_data=params["use_cached_data"],
        use_synthetic_data=params["use_synthetic_data"],
        num_datasets=num_datasets,
        params=params)

    eval_input_fn = dataloader.CriteoTsvReader(
        file_path=_csv_record_path("eval"),
        feature_config=feature_config,
        is_training=False,
        parallelism=16,
        use_cached_data=params["use_cached_data"],
        use_synthetic_data=params["use_synthetic_data"],
        num_datasets=num_datasets,
        params=params)

  return train_input_fn, eval_input_fn


def run_model(params,
              eval_init_fn=None,
              eval_finish_fn=None,
              run_finish_fn=None):
  """Run the DLRM model, using a pre-defined configuration.

  Args:
    params: HPTuner object that provides new params for the trial.
    eval_init_fn: Lambda to run at start of eval. None means use the default.
    eval_finish_fn: Lambda for end of eval. None means use the default.
    run_finish_fn: Lambda for end of execution. None means use the default.

  Returns:
    A list of tuples, each entry describing the eval metric for one eval. Each
    tuple entry is (global_step, metric_value).
  """
  mlp_log.mlperf_print(key="cache_clear", value=True)
  mlp_log.mlperf_print(key="init_start", value=None)
  mlp_log.mlperf_print("global_batch_size", params["batch_size"])
  mlp_log.mlperf_print("train_samples", _NUM_TRAIN_EXAMPLES)
  mlp_log.mlperf_print("eval_samples", _NUM_EVAL_EXAMPLES)
  adjusted_lr = params["learning_rate"] * (params["batch_size"] / 2048.0)
  mlp_log.mlperf_print("opt_base_learning_rate", adjusted_lr)
  mlp_log.mlperf_print("sgd_opt_base_learning_rate", adjusted_lr)
  mlp_log.mlperf_print("sgd_opt_learning_rate_decay_poly_power", 2)
  mlp_log.mlperf_print("sgd_opt_learning_rate_decay_steps",
                       params["decay_steps"])
  mlp_log.mlperf_print("lr_decay_start_steps", params["decay_start_step"])
  mlp_log.mlperf_print("opt_learning_rate_warmup_steps",
                       params["lr_warmup_steps"])
  mlp_log.mlperf_print("gradient_accumulation_steps", 1)
  seed = np.random.randint(10e8)
  tf.set_random_seed(seed)
  mlp_log.mlperf_print("seed", seed)

  # Used for REDACTED. List of tuples. Each entry is (global_step, auc_metric).
  eval_metrics = [(0, 0.0)]

  feature_config = fc.FeatureConfig(params)
  (feature_to_config_dict,
   table_to_config_dict) = feature_config.get_feature_tbl_config()
  opt_params = {
      "sgd":
          tpu_embedding.StochasticGradientDescentParameters(
              learning_rate=params["learning_rate"]),
      "adagrad":
          tpu_embedding.AdagradParameters(
              learning_rate=params["learning_rate"],
              initial_accumulator=params["adagrad_init_accum"])
  }
  embedding = None
  if table_to_config_dict and feature_to_config_dict:
    embedding = tpu_embedding.TPUEmbedding(
        table_to_config_dict,
        feature_to_config_dict,
        params["batch_size"],
        mode=tpu_embedding.TRAINING,
        optimization_parameters=opt_params[params["optimizer"]],
        partition_strategy=FLAGS.partition_strategy,
        pipeline_execution_with_tensor_core=FLAGS.pipeline_execution,
        master=FLAGS.master)

  num_hosts = FLAGS.num_tpu_shards // FLAGS.replicas_per_host
  if embedding:
    runner = dlrm_embedding_runner.DLRMEmbeddingRunner(
        iterations_per_loop=FLAGS.steps_between_evals,
        train_steps=FLAGS.train_steps,
        eval_steps=FLAGS.eval_steps,
        num_replicas=FLAGS.num_tpu_shards,
        num_outfeed_threads=num_hosts,
        device_order_mode=device_assignment.DeviceOrderMode.MESH,
        sparse_features_key="cat-features",
        embedding=embedding)
  else:
    runner = tr.TrainAndEvalRunner(
        iterations_per_loop=FLAGS.steps_between_evals,
        train_steps=FLAGS.train_steps,
        eval_steps=FLAGS.eval_steps,
        num_replicas=FLAGS.num_tpu_shards,
        device_order_mode=device_assignment.DeviceOrderMode.MESH)

  train_input_fn, eval_input_fn = get_input_fns(params, feature_config)

  input_fn_params = {
      "seed": seed
  }
  runner.initialize(
      train_input_fn,
      eval_input_fn,
      functools.partial(dlrm.dlrm_llr_model_fn, params, feature_config),
      params["batch_size"],
      params["eval_batch_size"],
      train_has_labels=False,
      eval_has_labels=False,
      params=input_fn_params
  )

  mlp_log.mlperf_print("init_stop", None)
  mlp_log.mlperf_print("run_start", None)

  num_evals = FLAGS.train_steps // FLAGS.steps_between_evals

  def _default_eval_init_fn(cur_step):
    """Logging statements executed before every eval."""
    eval_num = cur_step // FLAGS.steps_between_evals if FLAGS.steps_between_evals else 0
    tf.logging.info("== Block {}. Step {} of {}".format(eval_num + 1, cur_step,
                                                        FLAGS.train_steps))
    epoch_num = 1.0 + (1.0 * (eval_num + 1) / num_evals)
    mlp_log.mlperf_print(
        "block_start",
        None,
        metadata={
            "first_epoch_num": epoch_num,
            "epoch_count": 1
        })

  def _compute_metrics_async(in_q, out_q):
    success = False
    while True:
      q_item = in_q.get()
      if q_item is None:
        break
      if success:
        # Already reached threshold accuracy. Drop subsequent inputs.
        in_q.task_done()
        continue
      (cur_step, eval_output, summary_writer) = q_item
      eval_num = cur_step // FLAGS.steps_between_evals if FLAGS.steps_between_evals else 0
      epoch_num = 1.0 + (1.0 * (eval_num + 1) / num_evals)
      mlp_log.mlperf_print(
          "eval_start", None, metadata={"epoch_num": epoch_num})
      tf.logging.info(
          ".. [cm-{}] beginning _compute_metric".format(eval_num + 1))
      results_np = np.array(eval_output["results"])
      results_np = np.reshape(results_np, (-1, 2))
      predictions_np = results_np[:, 0].astype(np.float32)
      targets_np = results_np[:, 1].astype(np.int32)
      roc_obj = roc_metrics.RocMetrics(predictions_np, targets_np)
      roc_auc = roc_obj.ComputeRocAuc()
      mlp_log.mlperf_print(
          "eval_accuracy", roc_auc, metadata={"epoch_num": epoch_num})
      mlp_log.mlperf_print(
          "eval_stop", None, metadata={"epoch_num": epoch_num})
      success = roc_auc >= _ACCURACY_THRESH
      if success:
        mlp_log.mlperf_print("run_stop", None, metadata={"status": "success"})
      tf.logging.info(".. [cm-{}] Eval shape: {}.  AUC = {:.4f}".format(
          eval_num + 1, predictions_np.shape, roc_auc))
      if summary_writer:
        summary_writer.add_summary(
            utils.create_scalar_summary("auc", roc_auc),
            global_step=cur_step + FLAGS.steps_between_evals)
      eval_metrics.append((cur_step + FLAGS.steps_between_evals, roc_auc))
      in_q.task_done()
    out_q.put(success)

  def _async_eval_finish_fn(cur_step, eval_output, summary_writer=None):
    eval_num = cur_step // FLAGS.steps_between_evals if FLAGS.steps_between_evals else 0
    epoch_num = 1.0 + (1.0 * (eval_num + 1) / num_evals)
    mlp_log.mlperf_print(
        "block_stop", None, metadata={"first_epoch_num": epoch_num})
    eval_metric_q.put((cur_step, eval_output, summary_writer), block=False)
    return False

  def _default_run_finish_fn(success_status):
    del success_status
    runner.retrieve_embedding_vars()

  eval_metric_q = queue.Queue()
  result_q = queue.Queue()
  eval_metric_thread = threading.Thread(
      target=_compute_metrics_async, args=(eval_metric_q, result_q))
  eval_metric_thread.start()

  runner.train_and_eval(
      eval_init_fn=eval_init_fn or _default_eval_init_fn,
      eval_finish_fn=eval_finish_fn or _async_eval_finish_fn,
      run_finish_fn=run_finish_fn or _default_run_finish_fn)

  # Clean up eval metric offload queue and signal done to worker thread.
  eval_metric_q.join()
  eval_metric_q.put(None)

  success = result_q.get()
  assert result_q.empty(), (
      "Compute metrics thread unexpectedly returned more than one value.")
  if not success:
    mlp_log.mlperf_print("run_stop", None, metadata={"status": "failure"})

  eval_metric_thread.join()

  return eval_metrics


def main(argv):
  del argv
  params = common.get_params()
  run_model(params)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.disable_v2_behavior()
  common.define_dlrm_flags()
  absl_app.run(main)

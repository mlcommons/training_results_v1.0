import math
import os
import time
import copy

import mxnet
import numpy as np


def process_performance_stats(timestamps, batch_size, mode):
    """ Get confidence intervals

    :param timestamps: Collection of timestamps
    :param batch_size: Number of samples per batch
    :param mode: Estimator's execution mode
    :return: Stats
    """
    timestamps_ms = 1000 * timestamps
    throughput_imgps = (1000.0 * batch_size / timestamps_ms).mean()
    stats = {f"throughput_{mode}": throughput_imgps,
             f"latency_{mode}_mean": timestamps_ms.mean()}
    for level in [90, 95, 99]:
        stats.update({f"latency_{mode}_{level}": np.percentile(timestamps_ms, level)})

    return stats


def get_callbacks(flags, logger, eval_ranks, global_rank, world_size):
    samples_per_cycle = math.ceil(168/(flags.batch_size * world_size))
    samples_per_cycle *= flags.batch_size * world_size * flags.evaluate_every
    callbacks = []
    if global_rank == eval_ranks[0]:
        # For sync run train_ranks == eval_ranks. For async, we do not care about throughput
        if flags.benchmark:
            callbacks.append(PerformanceCallback(logger,
                                                 samples=samples_per_cycle,
                                                 warmup_steps=flags.warmup_steps,
                                                 max_steps=flags.start_eval_at // flags.evaluate_every,
                                                 mode='train'))
        else:
            callbacks.append(EvaluationCallback(logger, metric="mean_dice", seed=flags.seed,
                                                threshold=flags.quality_threshold))
        if flags.save_ckpt_path:
            callbacks.append(CheckpointCallback(flags.save_ckpt_path, metric="mean_dice"))

    return callbacks


class BaseCallback:
    def on_fit_start(self, **kwargs):
        pass

    def on_cycle_start(self, **kwargs):
        pass

    def on_cycle_end(self, **kwargs):
        pass

    def on_fit_end(self, **kwargs):
        pass


class PerformanceCallback(BaseCallback):
    def __init__(self, logger, samples, warmup_steps=1, max_steps=2, mode='train'):
        self._logger = logger
        self._samples = samples
        self._warmup_steps = warmup_steps
        self._max_steps = max_steps
        self._step = 0
        self._timestamps = []
        self._mode = mode

    def on_cycle_start(self, *args, **kwargs):
        self._step += 1
        if self._step >= self._warmup_steps and self._step < self._max_steps:
            self._timestamps.append(time.time())

    def on_fit_end(self, **kwargs):
        deltas = np.array([self._timestamps[i + 1] - self._timestamps[i] for i in range(len(self._timestamps) - 1)])
        try:
            stats = process_performance_stats(deltas, self._samples, self._mode)
        except IndexError:
            stats = {"throughput": 0}

        self._logger.log(step=(), data=stats)
        self._logger.flush()


class EvaluationCallback(BaseCallback):
    def __init__(self, logger, metric, threshold=0.908, seed=0):
        self._logger = logger
        self._best_metrics = {}
        self._initialized = False
        self._main_metric = metric
        self._prefix = "TOP_"
        self._last_epoch = 0
        self._first_epoch_above_threshold = 0
        self._threshold = threshold
        self._seed = seed
        self._training_start_time = None

    def on_fit_start(self, **kwargs):
        self._training_start_time = time.time()

    def on_cycle_end(self, epoch, metrics, *args, **kwargs):
        if not self._initialized:
            self._register_metrics(metrics)
        if self._best_metrics[self._prefix + self._main_metric] < metrics[self._main_metric]:
            for key in metrics.keys():
                self._best_metrics[self._prefix + key] = float(metrics[key])

        if metrics[self._main_metric] >= self._threshold and self._first_epoch_above_threshold == 0:
            self._first_epoch_above_threshold = epoch

        for key in metrics.keys():
            metrics[key] = float(metrics[key])
        self._last_epoch = epoch
        self._logger.log(step=(metrics["epoch"]), data={**metrics, **self._best_metrics})
        self._logger.flush()

    def _register_metrics(self, metrics):
        for key in metrics.keys():
            self._best_metrics[self._prefix + key] = float(metrics[key])
        self._initialized = True

    def on_fit_end(self, **kwargs):
        self._best_metrics["last_epoch"] = self._last_epoch
        self._best_metrics["first_conv_ep"] = self._first_epoch_above_threshold
        self._best_metrics["seed"] = self._seed
        self._best_metrics["total_time"] = (time.time() - self._training_start_time) / 60
        self._logger.log(step=(), data=self._best_metrics)
        self._logger.flush()


class CheckpointCallback(BaseCallback):
    def __init__(self, path, metric):
        self._path = path
        self._main_metric = metric
        self._best_metric = 0.0
        self._best_state = {}
        self._last_state = {}

    def on_cycle_end(self, epoch, metrics, model, **kwargs):
        if metrics[self._main_metric] > self._best_metric:
            self._best_metric = metrics[self._main_metric]
            model.save_parameters(os.path.join(self._path, "best_model.params"))

    def on_fit_end(self, model, **kwargs):
        model.save_parameters(os.path.join(self._path, "last_model.params"))

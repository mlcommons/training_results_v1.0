import os
import subprocess
import collections

import numpy as np
# import horovod.mxnet as hvd

import dllogger as logger
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend
from mlperf_logging import mllog
from mlperf_logging.mllog import constants


class MPIWrapper(object):
    def __init__(self):
        self.comm = None
        self.MPI = None

    def get_comm(self):
        if self.comm is None:
            import mpi4py
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI = MPI

        return self.comm

    def barrier(self):
        c = self.get_comm()
        # NOTE: MPI_Barrier is *not* working reliably at scale. Using MPI_Allreduce instead.
        #c.Barrier()
        val = np.ones(1, dtype=np.int32)
        result = np.zeros(1, dtype=np.int32)
        c.Allreduce(val, result)

    def allreduce(self, x):
        c = self.get_comm()
        rank = c.Get_rank()
        val = np.array(x, dtype=np.int32)
        result = np.zeros_like(val, dtype=np.int32)
        c.Allreduce([val, self.MPI.INT], [result, self.MPI.INT]) #, op=self.MPI.SUM)
        return result

    def rank(self):
        c = self.get_comm()
        return c.Get_rank()

CONSTANTS = constants
mllogger = mllog.get_mllogger()
mpiwrapper = MPIWrapper()


def get_logger(params, eval_ranks, global_rank):
    backends = []
    if global_rank == eval_ranks[0]:
        if params.verbose:
            backends += [StdOutBackend(Verbosity.VERBOSE)]
        if params.log_dir:
            backends += [JSONStreamBackend(Verbosity.VERBOSE, params.log_dir)]
    logger.init(backends=backends)
    return logger


def get_mlperf_logger(path, filename='mlperf.log'):
    mllog.config(filename=os.path.join(path, filename))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    return mllogger


def mllog_start(*args, **kwargs):
    _mllog_print(mllogger.start, *args, **kwargs)


def mllog_end(*args, **kwargs):
    _mllog_print(mllogger.end, *args, **kwargs)


def mllog_event(*args, **kwargs):
    _mllog_print(mllogger.event, *args, **kwargs)


def _mllog_print(logger, *args, **kwargs):
    """
    Wrapper for MLPerf compliance logging calls.
    All arguments but 'sync' are passed to mlperf_log.mllog_print function.
    If 'sync' is set to True then the wrapper will synchronize all distributed
    workers. 'sync' should be set to True for all compliance tags that require
    accurate timing (RUN_START, RUN_STOP etc.)
    """
    if kwargs.pop('sync', False):
        mpiwrapper.barrier()
    if 'value' not in kwargs:
        kwargs['value'] = None
    force = kwargs.pop('force', False)
    if mpiwrapper.rank() == 0 or force:
        logger(*args, **kwargs, stack_offset=3)


def mlperf_submission_log(num_nodes=1):
    mllog_event(
        key=mllog.constants.SUBMISSION_BENCHMARK,
        value=constants.UNET3D,
    )

    mllog_event(
        key=mllog.constants.SUBMISSION_ORG,
        value='nvidia')

    mllog_event(
        key=mllog.constants.SUBMISSION_DIVISION,
        value='closed')

    mllog_event(
        key=mllog.constants.SUBMISSION_STATUS,
        value='onprem')

    mllog_event(
        key=mllog.constants.SUBMISSION_PLATFORM,
        value=f'{num_nodes}xSUBMISSION_PLATFORM_PLACEHOLDER')


def mlperf_run_param_log(flags):
    mllog_event(key=mllog.constants.OPT_NAME, value=flags.optimizer)
    mllog_event(key=mllog.constants.OPT_BASE_LR, value=flags.learning_rate)
    mllog_event(key=mllog.constants.OPT_LR_WARMUP_EPOCHS, value=flags.lr_warmup_epochs)
    # mllog_event(key=mllog.constants.OPT_LR_WARMUP_FACTOR, value=flags.lr_warmup_factor)
    mllog_event(key=mllog.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=flags.lr_decay_epochs)
    mllog_event(key=mllog.constants.OPT_LR_DECAY_FACTOR, value=flags.lr_decay_factor)
    mllog_event(key=mllog.constants.OPT_WEIGHT_DECAY, value=flags.weight_decay)
    mllog_event(key="opt_momentum", value=flags.momentum)
    mllog_event(key="oversampling", value=flags.oversampling)
    mllog_event(key="training_input_shape", value=flags.input_shape)
    mllog_event(key="validation_input_shape", value=flags.val_input_shape)
    mllog_event(key="validation_overlap", value=flags.overlap)


# def log_env_info():
#     """
#     Prints information about execution environment.
#     """
#     print('Collecting environment information...')
#     env_info = torch.utils.collect_env.get_pretty_env_info()
#     print(f'{env_info}')

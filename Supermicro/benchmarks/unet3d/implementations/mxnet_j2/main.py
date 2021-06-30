import math
import os
import shutil
import numpy as np

import mxnet as mx
from mxnet.contrib import amp
import horovod.mxnet as hvd
from mpi4py import MPI

from mlperf_logging import mllog
from mlperf_logging.mllog import constants
from mlperf_logger import get_logger, mllog_start, mllog_end, mllog_event, mlperf_run_param_log

from model.unet3d import Unet3D
from model.losses import DiceScore
from data_loading.data_loader import get_data_loaders, get_dummy_loaders
from runtime.arguments import PARSER
from runtime.training import train
from runtime.warmup import train as train_init
from runtime.distributed import distribute_mpiranks, get_group_comm
from runtime.inference import evaluate, SlidingWindow
from runtime.setup import seed_everything, get_seed, cleanup_scratch_space, get_rnd_scratch_space
from runtime.callbacks import get_callbacks


def main():
    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet3d.log'))
    mllog.config(filename=os.path.join("/results", 'unet3d.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    mllog_event(key=constants.CACHE_CLEAR, value=True)
    mllog_start(key=constants.INIT_START)

    flags = PARSER.parse_args()
    comm = MPI.COMM_WORLD
    global_rank = comm.Get_rank()
    world_size = comm.Get_size()
    local_rank = global_rank % flags.gpu_per_node
    local_size = flags.gpu_per_node

    train_ranks, eval_ranks, transfer_ranks = distribute_mpiranks(local_rank, local_size, world_size,
                                                                  flags.nodes_for_eval, flags.gpu_per_node)

    train_comm = get_group_comm(comm, train_ranks)
    eval_comm = get_group_comm(comm, eval_ranks)
    transfer_comm = get_group_comm(comm, transfer_ranks)

    scratch_space = get_rnd_scratch_space(flags.network_dir)
    cleanup_scratch_space(scratch_space, flags.nodes_for_eval, global_rank)

    worker_seed = flags.seed
    if global_rank in train_ranks:
        hvd.init(train_comm)
        worker_seed = get_seed(flags.seed, flags.spatial_group_size, flags.use_spatial_loader)
        seed_everything(worker_seed)
        print(f"RANK {hvd.rank()}. SEED {worker_seed}")
        mllog_event(key=constants.SEED, value=flags.seed if flags.seed != -1 else worker_seed, sync=False)

    ctx = mx.gpu(local_rank)
    if global_rank == 0:
        mlperf_run_param_log(flags)

    if flags.amp:
        amp.init()

    dllogger = get_logger(flags, eval_ranks, global_rank)
    callbacks = get_callbacks(flags, dllogger, eval_ranks, global_rank, world_size)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, include_background=flags.include_background)
    sw_inference = SlidingWindow(batch_size=flags.val_batch_size,
                                 mode="gaussian",
                                 roi_shape=flags.val_input_shape,
                                 precision=np.float16 if flags.static_cast or flags.amp else np.float32,
                                 data_precision=np.float16 if flags.static_cast else np.float32,
                                 ctx=ctx,
                                 local_rank=local_rank,
                                 use_spatial_loader=flags.use_spatial_loader,
                                 spatial_group_size=flags.spatial_group_size,
                                 cache_dataset=flags.cache_eval_dataset)

    current_comm = train_comm if global_rank in train_ranks else eval_comm
    model = Unet3D(n_classes=3, spatial_group_size=flags.spatial_group_size,
                   local_rank=local_rank, comm=current_comm, spatial_loader=flags.use_spatial_loader)

    global_batch_size = flags.batch_size * (len(train_ranks) // flags.spatial_group_size)
    steps_per_epoch = math.ceil(168/global_batch_size)
    world_size = world_size if global_rank in train_ranks else 1
    model.init(flags, ctx=ctx, world_size=world_size, steps_per_epoch=steps_per_epoch,
               is_training_rank=global_rank in train_ranks)
    if flags.static_cast:
        model.cast('float16')

    # if flags.exec_mode == 'train' and flags.warmup:
    #     train_loader, val_loader = get_dummy_loaders(flags, data_dir="/tmp/dummy_data", seed=worker_seed,
    #                                                  local_rank=local_rank, global_rank=global_rank,
    #                                                  training_ranks=train_ranks,
    #                                                  spatial_group_size=flags.spatial_group_size)
    #     train_init(flags, model, train_loader, comm, train_comm, eval_comm, transfer_comm,
    #                train_ranks, eval_ranks, ctx=ctx)
    #     model.initialize(ctx=ctx, force_reinit=True)
    #     print("MODEL WARMED UP")
    #     del train_loader, val_loader

    mllog_end(key=constants.INIT_STOP, sync=True)

    mllog_start(key=constants.RUN_START, sync=True)
    train_loader, val_loader = get_data_loaders(flags, data_dir=flags.data_dir, seed=worker_seed,
                                                local_rank=local_rank, global_rank=global_rank,
                                                train_ranks=train_ranks, eval_ranks=eval_ranks,
                                                spatial_group_size=flags.spatial_group_size)
    mllog_event(key=constants.GLOBAL_BATCH_SIZE, sync=False, value=global_batch_size)
    mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, sync=False, value=1)

    if flags.exec_mode == 'train':
        train(flags, model, train_loader, val_loader, score_fn, sw_inference,
              comm, train_comm, eval_comm, transfer_comm,
              train_ranks, eval_ranks, transfer_ranks, ctx=ctx, callbacks=callbacks)

        if global_rank == eval_ranks[0] and flags.nodes_for_eval:
            if os.path.exists(scratch_space):
                shutil.rmtree(scratch_space)

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx=ctx, eval_comm=eval_comm)
        eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx=ctx, eval_comm=eval_comm)
        if global_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])


if __name__ == '__main__':
    main()

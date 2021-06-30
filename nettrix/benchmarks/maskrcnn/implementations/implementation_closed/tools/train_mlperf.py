# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import functools
import logging
import random
import datetime
import time
import gc
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.tester import test
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.mlperf_logger import log_end, log_start, log_event, generate_seeds, broadcast_seeds, barrier, configure_logger
from maskrcnn_benchmark.utils.async_evaluator import init, get_evaluator, set_epoch_tag, get_tag
from maskrcnn_benchmark.utils.timed_section import TimedSection

from fp16_optimizer import FP16_Optimizer

from mlperf_logging.mllog import constants
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

torch.backends.cudnn.deterministic = True
# Loop over all finished async results, return a dict of { tag : (bbox_map, segm_map) }
finished_prep_work = None
def check_completed_tags(iteration):
    # Check for completeness is fairly expensive, so we only do it once per N iterations
    if iteration % 10 != 9:
        return {}

    global finished_prep_work
    from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults, all_gather_prep_work, evaluate_coco
    if get_world_size() > 1:
        num_finished = torch.zeros([1], dtype=torch.int32, device='cuda') if finished_prep_work is None else torch.ones([1], dtype=torch.int32, device='cuda')
        torch.distributed.all_reduce(num_finished)
        ready_to_submit_evaluation_task = True if num_finished == torch.distributed.get_world_size() else False
    else:
        ready_to_submit_evaluation_task = False if finished_prep_work is None else True
    evaluator = get_evaluator()
    if ready_to_submit_evaluation_task:
        with TimedSection("EXPOSED: Launching evaluation task took %.3fs"):
            coco_results, iou_types, coco, output_folder = finished_prep_work
            finished_prep_work = None
            coco_results = all_gather_prep_work(coco_results)
            if is_main_process():
                evaluator.submit_task(get_tag(),
                                      evaluate_coco,
                                      coco,
                                      coco_results,
                                      iou_types,
                                      output_folder)
    else:
        # loop over all all epoch, result pairs that have finished
        all_results = {}
        for t, r in evaluator.finished_tasks().items():
            # Note: one indirection due to possibility of multiple test datasets
            # we only care about the first
            map_results = r# [0]
            if isinstance(map_results, COCOResults):
                bbox_map = map_results.results["bbox"]['AP']
                segm_map = map_results.results["segm"]['AP']
                all_results.update({ t : (bbox_map, segm_map) })
            else:
                finished_prep_work = map_results

        return all_results

    return {}

def mlperf_test_early_exit(iteration, iters_per_epoch, tester, model, distributed, min_bbox_map, min_segm_map):
    # Note: let iters / epoch == 10k, at iter 9999 we've finished epoch 0 and need to test
    if iteration > 0 and (iteration + 1)% iters_per_epoch == 0:
        synchronize()
        epoch = iteration // iters_per_epoch + 1

        log_end(key=constants.EPOCH_STOP, metadata={"epoch_num": epoch})
        log_end(key=constants.BLOCK_STOP, metadata={"first_epoch_num": epoch})
        log_start(key=constants.EVAL_START, metadata={"epoch_num":epoch})
        # set the async evaluator's tag correctly
        set_epoch_tag(epoch)

        # Note: No longer returns anything, underlying future is in another castle
        tester(model=model, distributed=distributed)
        # necessary for correctness
        model.train()
    elif iteration % 10 == 9: # do finished check after every 10 iterations
        # Otherwise, check for finished async results
        results = check_completed_tags(iteration)

        # on master process, check each result for terminating condition
        # sentinel for run finishing
        finished = 0
        if is_main_process():
            for result_epoch, (bbox_map, segm_map) in results.items():
                logger = logging.getLogger('maskrcnn_benchmark.trainer')
                logger.info('bbox mAP: {}, segm mAP: {}'.format(bbox_map, segm_map))

                log_event(key=constants.EVAL_ACCURACY, value={"BBOX" : bbox_map, "SEGM" : segm_map}, metadata={"epoch_num" : result_epoch} )
                log_end(key=constants.EVAL_STOP, metadata={"epoch_num": result_epoch})
                # terminating condition
                if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
                    logger.info("Target mAP reached, exiting...")
                    finished = 1
                    #return True

        # We now know on rank 0 whether or not we should terminate
        # Bcast this flag on multi-GPU
        if get_world_size() > 1:
            with torch.no_grad():
                finish_tensor = torch.tensor([finished], dtype=torch.int32, device = torch.device('cuda'))
                torch.distributed.broadcast(finish_tensor, 0)

                # If notified, end.
                if finish_tensor.item() == 1:
                    return True
        else:
            # Single GPU, don't need to create tensor to bcast, just use value directly
            if finished == 1:
                return True

    # Otherwise, default case, continue
    return False

def mlperf_log_epoch_start(iteration, iters_per_epoch):
    # First iteration:
    #     Note we've started training & tag first epoch start
    if iteration == 0:
        log_start(key=constants.BLOCK_START, metadata={"first_epoch_num":1, "epoch_count":1})
        log_start(key=constants.EPOCH_START, metadata={"epoch_num":1})
        return
    if iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch + 1
        log_start(key=constants.BLOCK_START, metadata={"first_epoch_num": epoch, "epoch_count": 1})
        log_start(key=constants.EPOCH_START, metadata={"epoch_num": epoch})

from maskrcnn_benchmark.layers.batch_norm import FrozenBatchNorm2d
from maskrcnn_benchmark.layers.nhwc.batch_norm import FrozenBatchNorm2d_NHWC
from maskrcnn_benchmark.modeling.backbone.resnet import Bottleneck
def cast_frozen_bn_to_half(module):
    if isinstance(module, FrozenBatchNorm2d) or isinstance(module, FrozenBatchNorm2d_NHWC):
        module.half()
    for child in module.children():
        cast_frozen_bn_to_half(child)
    return module


def train(cfg, local_rank, distributed, random_number_generator=None, seed=None):

    # Model logging
    log_event(key=constants.GLOBAL_BATCH_SIZE, value=cfg.SOLVER.IMS_PER_BATCH)
    log_event(key=constants.NUM_IMAGE_CANDIDATES, value=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN)
    log_event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=1)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # Initialize mixed-precision training
    is_fp16 = (cfg.DTYPE == "float16")
    if is_fp16:
        # convert model to FP16
        model.half()

    # - CUDA graph ------
    from function import graph

    if cfg.USE_CUDA_GRAPH:

        images_per_gpu = cfg.SOLVER.IMS_PER_BATCH // get_world_size()
        print("USE_CUDA_GRAPH :: images_per_gpu = %d" % (images_per_gpu))

        min_size = cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MIN_SIZE_TRAIN, tuple) else cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MAX_SIZE_TRAIN, tuple) else cfg.INPUT.MAX_SIZE_TRAIN
        divisibility = max(1, cfg.DATALOADER.SIZE_DIVISIBILITY)
        shapes_per_orientation = cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION

        min_size = ((min_size + divisibility - 1) // divisibility) * divisibility
        max_size = ((max_size + divisibility - 1) // divisibility) * divisibility
        size_range = (max_size - min_size) // divisibility

        shapes = []
        for i in range(0,shapes_per_orientation):
            size = min_size + ((i+1) * size_range // shapes_per_orientation) * divisibility
            shapes.append( (min_size, size) )
            shapes.append( (size, min_size) )
        print(shapes)

        graphed_forwards = {}
        graph_stream = torch.cuda.Stream()
        for i, shape in enumerate(shapes):
            dummy_shape = (images_per_gpu,) + shape + (3,) if cfg.NHWC else (images_per_gpu,3,) + shape
            dummy_batch = torch.ones(dummy_shape, dtype=torch.float16, device=device)
            dummy_image_sizes = torch.tensor([list(shape) for _ in range(images_per_gpu)], dtype=torch.float32, device=device)
            sample_args = (dummy_batch.clone(),dummy_image_sizes.clone(),)

            forward_fn = "graph_forward_%d" % (i+1)
            if i == 0:
                model.graphable = graph(model.graphable,
                                       sample_args,
                                       graph_stream=graph_stream,
                                       warmup_only=True,
                                       overwrite_fn='eager_forward')
                model.graphable, pool_id = graph(model.graphable,
                                                sample_args,
                                                graph_stream=graph_stream,
                                                warmup_only=False,
                                                overwrite_fn=forward_fn,
                                                return_pool_id=True)
            else:
                model.graphable = graph(model.graphable,
                                       sample_args,
                                       graph_stream=graph_stream,
                                       warmup_only=False,
                                       overwrite_fn=forward_fn,
                                       use_pool_id=pool_id)
            graphed_forwards[dummy_shape] = getattr(model.graphable, forward_fn)

        class GraphedWrapper(torch.nn.Module):
            def __init__(self, model_segment, graphed_forwards):
                super().__init__()
                self.model_segment = model_segment
                self.graphed_forwards = graphed_forwards

            def forward(self, images_tensor, image_sizes_tensor):
                shape = tuple(list(images_tensor.shape))
                if shape in self.graphed_forwards:
                    return self.graphed_forwards[shape](images_tensor, image_sizes_tensor)
                elif images_tensor.shape[0] < images_per_gpu:
                    # run with padding in case of in-complete batch
                    # pad
                    before_pad = images_tensor.shape[0]
                    images_tensor = torch.nn.functional.pad(images_tensor, (0,0,0,0,0,0,0,images_per_gpu-before_pad))
                    image_sizes_tensor = torch.nn.functional.pad(image_sizes_tensor, (0,0,0,images_per_gpu-before_pad))
                    # run with graph
                    shape = tuple(list(images_tensor.shape))
                    if shape in self.graphed_forwards:
                        out = self.graphed_forwards[shape](images_tensor, image_sizes_tensor)
                    else:
                        out = self.model_segment.eager_forward(images_tensor, image_sizes_tensor)
                    # unpad
                    out = [o[0:before_pad] for o in out]
                    return out
                else:
                    return self.model_segment.eager_forward(images_tensor, image_sizes_tensor)

        model.graphable = GraphedWrapper(model.graphable, graphed_forwards)
    else:
        shapes = None
    # ------------------

    optimizer = make_optimizer(cfg, model)
    # Optimizer logging
    log_event(key=constants.OPT_NAME, value="sgd_with_momentum")
    log_event(key=constants.OPT_BASE_LR, value=cfg.SOLVER.BASE_LR)
    log_event(key=constants.OPT_LR_WARMUP_STEPS, value=cfg.SOLVER.WARMUP_ITERS)
    log_event(key=constants.OPT_LR_WARMUP_FACTOR, value=cfg.SOLVER.WARMUP_FACTOR)
    log_event(key=constants.OPT_LR_DECAY_FACTOR, value=cfg.SOLVER.GAMMA)
    log_event(key=constants.OPT_LR_DECAY_STEPS, value=cfg.SOLVER.STEPS)
    log_event(key=constants.MIN_IMAGE_SIZE, value=cfg.INPUT.MIN_SIZE_TRAIN[0])
    log_event(key=constants.MAX_IMAGE_SIZE, value=cfg.INPUT.MAX_SIZE_TRAIN)

    scheduler = make_lr_scheduler(cfg, optimizer)

    # disable the garbage collection
    gc.disable()

    if distributed:
        model = DDP(model, delay_allreduce=True)
        numels = 0
        for p in model.parameters():
            numels += p.numel()
        print("model has %d elements of size %d bytes" % (numels, 2 if is_fp16 else 4))

    arguments = {}
    arguments["iteration"] = 0
    arguments["nhwc"] = cfg.NHWC
    arguments['ims_per_batch'] = cfg.SOLVER.IMS_PER_BATCH
    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    arguments["save_checkpoints"] = cfg.SAVE_CHECKPOINTS

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, cfg.NHWC)
    arguments.update(extra_checkpoint_data)

    if is_fp16:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True, dynamic_loss_scale_window=cfg.DYNAMIC_LOSS_SCALE_WINDOW)

    log_end(key=constants.INIT_STOP)
    barrier()
    log_start(key=constants.RUN_START)
    barrier()

    data_loader, iters_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        random_number_generator=random_number_generator,
        seed=seed,
        shapes=shapes
    )
    log_event(key=constants.TRAIN_SAMPLES, value=len(data_loader))

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # set the callback function to evaluate and potentially
    # early exit each epoch
    if cfg.PER_EPOCH_EVAL:
        per_iter_callback_fn = functools.partial(
                mlperf_test_early_exit,
                iters_per_epoch=iters_per_epoch,
                tester=functools.partial(test, cfg=cfg, shapes=shapes),
                model=model,
                distributed=distributed,
                min_bbox_map=cfg.MLPERF.MIN_BBOX_MAP,
                min_segm_map=cfg.MLPERF.MIN_SEGM_MAP)
    else:
        per_iter_callback_fn = None

    start_train_time = time.time()

    success = do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        cfg.DISABLE_REDUCED_LOGGING,
        cfg.DISABLE_LOSS_LOGGING,
        per_iter_start_callback_fn=functools.partial(mlperf_log_epoch_start, iters_per_epoch=iters_per_epoch),
        per_iter_end_callback_fn=per_iter_callback_fn
    )

    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print(
            "&&&& MLPERF METRIC THROUGHPUT={:.4f} iterations / s".format((arguments["iteration"] * cfg.SOLVER.IMS_PER_BATCH) / total_training_time)
    )

    return model, success



def main():

    configure_logger(constants.MASKRCNN)
    log_start(key=constants.INIT_START)

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    # if is_main_process:
    #     # Setting logging file parameters for compliance logging
    #     os.environ["COMPLIANCE_FILE"] = '/MASKRCNN_complVv0.5.0_' + str(datetime.datetime.now())
    #     constants.LOG_FILE = os.getenv("COMPLIANCE_FILE")
    #     constants._FILE_HANDLER = logging.FileHandler(constants.LOG_FILE)
    #     constants._FILE_HANDLER.setLevel(logging.DEBUG)
    #     constants.LOGGER.addHandler(constants._FILE_HANDLER)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        # setting seeds - needs to be timed, so after RUN_START
        if is_main_process():
            master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
            seed_tensor = torch.tensor(master_seed, dtype=torch.float32, device=torch.device("cuda"))
        else:
            seed_tensor = torch.tensor(0, dtype=torch.float32, device=torch.device("cuda"))

        torch.distributed.broadcast(seed_tensor, 0)
        master_seed = int(seed_tensor.item())
    else:
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)

    # actually use the random seed
    args.seed = master_seed
    # random number generator with seed set to master_seed
    random_number_generator = random.Random(master_seed)
    log_event(key=constants.SEED, value=master_seed)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(random_number_generator, torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

    # todo sharath what if CPU
    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device='cuda')

    # Setting worker seeds
    logger.info("Worker {}: Setting seed {}".format(args.local_rank, worker_seeds[args.local_rank]))
    torch.manual_seed(worker_seeds[args.local_rank])
    random.seed(worker_seeds[args.local_rank])
    np.random.seed(worker_seeds[args.local_rank])

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Initialise async eval
    init()

    log_event(key='d_batch_size', value=cfg.SOLVER.IMS_PER_BATCH/num_gpus)

    model, success = train(cfg, args.local_rank, args.distributed, random_number_generator, seed=master_seed)

    if success is not None:
        if success:
            log_end(key=constants.RUN_STOP, metadata={"status": "success"})
        else:
            log_end(key=constants.RUN_STOP, metadata={"status": "aborted"})

if __name__ == "__main__":
    start = time.time()
    torch.set_num_threads(1)
    main()
    print("&&&& MLPERF METRIC TIME=", time.time() - start)

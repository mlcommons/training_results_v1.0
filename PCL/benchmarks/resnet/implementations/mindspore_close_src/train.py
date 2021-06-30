# Copyright 2021 PCL & PKU
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import argparse
import numpy as np
import time

from mindspore import context
from mindspore import Tensor
from mindspore.nn.optim import LARS, Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
import mindspore.dataset as ds
import mindspore.dataset.engine as de

from mlperf_logging import mllog
from dataset import create_dataset
from lr_generator import get_lr
from resnet import resnet50
from metric import DistAccuracy, ClassifyCorrectCell
from callback import StateMonitor
from cross_entropy import CrossEntropySmooth
from cfg_parser import merge_args

import moxing as mox


os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')

device_id = int(os.getenv('DEVICE_ID'))   # 0 ~ 7
local_rank = int(os.getenv('RANK_ID'))    # local_rank
device_num = int(os.getenv('RANK_SIZE'))  # world_size
log_filename = os.path.join(os.getcwd(), "resnet50_rank"+ str(local_rank) +".log")


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)
context.set_context(device_id=device_id)


def parse_args():
    parser = argparse.ArgumentParser(description='Image classification')

    # cloud
    parser.add_argument('--data_url', type=str, default=None, help='data_url')
    parser.add_argument('--train_url', type=str, default='./', help='train_url')

    # train datasets
    parser.add_argument('--dataset_path', type=str, default='/opt/npu/datasets/imagenet/train', help='Dataset path')
    parser.add_argument('--train_image_size', type=int, default=224, help='train_image_size')
    parser.add_argument('--crop_min', type=float, default=0.08, help='Dataset path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--train_num_workers', type=int, default=12, help='train_num_workers')

    # eval datasets
    parser.add_argument('--eval_path', type=str, default='/opt/npu/datasets/imagenet/val', help='Eval dataset path')
    parser.add_argument('--eval_image_size', type=int, default=224, help='eval_image_size')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='eval_batch_size')
    parser.add_argument('--eval_interval', type=int, default=4, help='eval_interval')
    parser.add_argument('--eval_offset', type=int, default=-1, help='1 means 4*n+1 epochs')
    parser.add_argument('--eval_num_workers', type=int, default=12, help='eval_num_workers')

    # network
    parser.add_argument('--backbone', type=str, default='resnet50', help='resnet50')
    parser.add_argument('--class_num', type=int, default=1001, help='class_num')
    parser.add_argument('--conv_init_mode', type=str, default='truncnorm', help='truncnorm/HeNormal/HeUniform')
    parser.add_argument('--bn_init_mode', type=str, default='adv_bn_init', help='adv_bn_init/conv_bn_init')

    # lr
    parser.add_argument('--lr_decay_mode', type=str, default='poly', help='lr_decay_mode')
    parser.add_argument('--poly_power', type=float, default=2, help='lars_opt_learning_rate_decay_poly_power')
    parser.add_argument('--lr_init', type=float, default=0.0, help='lr_init')
    parser.add_argument('--lr_max', type=float, default=0.8, help='lr_max')
    parser.add_argument('--lr_min', type=float, default=0.0, help='lr_min')
    parser.add_argument('--max_epoch', type=int, default=33, help='max_epoch')
    parser.add_argument('--warmup_epochs', type=float, default=1, help='warmup_epochs')

    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight_decay')
    parser.add_argument('--use_nesterov', type=int, default=0, help='use_nesterov')
    parser.add_argument('--use_lars', type=int, default=0, help='use_lars')
    parser.add_argument('--lars_epsilon', type=float, default=0.0, help='lars_epsilon')
    parser.add_argument('--lars_coefficient', type=float, default=0.001, help='lars_coefficient')

    # loss
    parser.add_argument('--loss_scale', type=int, default=1024, help='loss_scale')
    parser.add_argument('--use_label_smooth', type=int, default=1, help='use_label_smooth')
    parser.add_argument('--label_smooth_factor', type=float, default=0.1, help='label_smooth_factor')

    # args_yml_fn
    parser.add_argument('--args_yml_fn', type=str, default='', help='args_yml_fn')

    # seed
    parser.add_argument('--seed', type=int, default=1, help='seed')

    # gradient_accumulation_steps, set to '1' for resnet
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')

    args = parser.parse_args()
    args = merge_args(args, args.args_yml_fn)

    args.use_nesterov = (args.use_nesterov == 1)
    args.weight_decay = float(args.weight_decay)

    if args.eval_offset < 0:
        args.eval_offset = args.max_epoch % args.eval_interval

    args.dataset_path = "/cache_mlperf/imagenet/train"
    args.eval_path = "/cache_mlperf/imagenet/val"

    return args


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)

    context.set_auto_parallel_context(device_num=device_num,
                                      parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True)

    # mllog
    mllog.config(filename=log_filename)
    mllog.config(
        default_namespace="mindspore",
        default_stack_offset=1,
        default_clear_line=False,
        root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
    mllogger = mllog.get_mllogger()

    # submission
    mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value="resnet")
    mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value="closed")
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value="PCL & PKU")
    mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value="Ascend 910 ProA")
    mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value="cloud")
    mllogger.event(key=mllog.constants.CACHE_CLEAR)

    # init the distribute env
    init()

    # network
    net = resnet50(backbone=args.backbone,
                   class_num=args.class_num,
                   conv_init_mode=args.conv_init_mode,
                   bn_init_mode=args.bn_init_mode)

    # loss
    if not args.use_label_smooth:
        args.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True,
                              reduction="mean",
                              smooth_factor=args.label_smooth_factor,
                              num_classes=args.class_num)

    # train dataset
    epoch_size = args.max_epoch
    dataset = create_dataset(dataset_path=args.dataset_path,
                             do_train=True,
                             image_size=args.train_image_size,
                             crop_min=args.crop_min,
                             batch_size=args.batch_size,
                             num_workers=args.train_num_workers)
    ds.config.set_seed(args.seed)
    de.config.set_prefetch_size(64)

    step_size = dataset.get_dataset_size()
    args.steps_per_epoch = step_size

    # evalutation dataset
    eval_dataset = create_dataset(dataset_path=args.eval_path,
                                  do_train=False,
                                  image_size=args.eval_image_size,
                                  batch_size=args.eval_batch_size,
                                  num_workers=args.eval_num_workers)
    eval_step_size = eval_dataset.get_dataset_size()

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)

    # loss scale
    loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    # learning rate
    lr_array = get_lr(global_step=0, lr_init=args.lr_init, lr_end=args.lr_min, lr_max=args.lr_max,
                      warmup_epochs=args.warmup_epochs, total_epochs=epoch_size, steps_per_epoch=step_size,
                      lr_decay_mode=args.lr_decay_mode, poly_power=args.poly_power)
    lr = Tensor(lr_array)

    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': args.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]

    opt = Momentum(group_params, lr, args.momentum, loss_scale=args.loss_scale, use_nesterov=args.use_nesterov)
    if args.use_lars:
        opt = LARS(opt, epsilon=args.lars_epsilon, coefficient=args.lars_coefficient,
                   lars_filter=lambda x: 'beta' not in x.name and 'gamma' not in x.name and 'bias' not in x.name)

    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2",
                  keep_batchnorm_fp32=False,
                  metrics={'acc': DistAccuracy(batch_size=args.eval_batch_size, device_num=device_num)},
                  eval_network=dist_eval_network, 
                  total_steps=args.steps_per_epoch*args.max_epoch) 

    # set event
    mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=args.batch_size * device_num)
    mllogger.event(key="opt_name", value="lars")
    mllogger.event(key="lars_opt_base_learning_rate", value=args.lr_max)
    mllogger.event(key="lars_opt_end_learning_rate", value=args.lr_min)
    mllogger.event(key="lars_opt_learning_rate_decay_poly_power", value=args.poly_power)
    mllogger.event(key="lars_opt_learning_rate_decay_steps", value=step_size * (epoch_size - args.warmup_epochs))
    mllogger.event(key="lars_epsilon", value=args.lars_epsilon)
    mllogger.event(key="lars_opt_learning_rate_warmup_epochs", value=args.warmup_epochs)
    mllogger.event(key="lars_opt_momentum", value=args.momentum)
    mllogger.event(key="lars_opt_weight_decay", value=args.weight_decay)
    mllogger.event(key="gradient_accumulation_steps", value=args.gradient_accumulation_steps)
    mllogger.event(key="seed", value=args.seed)

    state_cb = StateMonitor(data_size=step_size,
                            mllogger=mllogger,
                            tot_batch_size=args.batch_size * device_num,
                            lrs=lr_array,
                            model=model,
                            eval_dataset=eval_dataset,
                            eval_interval=args.eval_interval,
                            eval_offset=args.eval_offset)

    cb = [state_cb, ]

    # compile
    mllogger.start(key=mllog.constants.INIT_START)
    model._init(dataset, eval_dataset, sink_size=step_size, epoch=epoch_size)
    mllogger.end(key=mllog.constants.INIT_STOP)

    sync_path = os.path.join(args.train_url, "sync_compile")
    if not mox.file.exists(sync_path):
        mox.file.make_dirs(sync_path)

    yml_name = os.path.splitext(os.path.split(args.args_yml_fn)[-1])[0]
    s3_rank_ready_file = os.path.join(sync_path, '{}_{}.txt'.format(yml_name, local_rank))
    if mox.file.exists(s3_rank_ready_file):
        mox.file.remove(s3_rank_ready_file, recursive=False)
        time.sleep(10)

    mox.file.write(s3_rank_ready_file, '{}'.format(local_rank))

    while local_rank == 0:
        existed = []
        all_rank_exist = True
        for rank_item in range(device_num):
            if rank_item not in existed:
                rank_fn_item = os.path.join(sync_path, '{}_{}.txt'.format(yml_name, rank_item))
                if not mox.file.exists(rank_fn_item):
                    print("rank_fn_item:{} is not exist".format(rank_fn_item))
                    all_rank_exist = False
                    break
                else:
                    existed.append(rank_item)

        if all_rank_exist:
            break
        else:
            time.sleep(1)

    # train and eval
    mllogger.start(key=mllog.constants.RUN_START)
    mllogger.event(key="train_samples", value=step_size*device_num*args.batch_size)
    mllogger.event(key="eval_samples", value=eval_step_size*device_num*args.eval_batch_size)
    model.train(epoch_size, dataset, callbacks=cb, sink_size=step_size, eval_interval=args.eval_interval)
    mllogger.event(key=mllog.constants.RUN_STOP, metadata={"status": "success"})

    # copy mllog
    src = log_filename
    mllog_dir = os.path.join(args.train_url, "mllog")
    if not mox.file.exists(mllog_dir):
        mox.file.make_dirs(mllog_dir)
    dst = os.path.join(mllog_dir, "resnet50_mllog_rank_{}.log".format(local_rank))
    mox.file.copy(src, dst)

# Copyright 2020 PCL & PKU
#
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
"""
#################pre_train bert example on zh-wiki########################
python run_pretrain.py
"""

import os
import ast
import time
import math
import argparse
from mindspore.profiler import Profiler
import mindspore.communication.management as D
from mindspore.communication.management import get_rank
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.train_thor import ConvertModelUtils
from mindspore.nn.optim import Lamb, Momentum, AdamWeightDecay, THOR
from mindspore import log as logger
from mindspore.common import set_seed
from src import BertNetworkWithLoss, BertTrainOneStepCell, BertTrainOneStepWithLossScaleCell, \
                BertTrainAccumulationAllReduceEachWithLossScaleCell, \
                BertTrainAccumulationAllReducePostWithLossScaleCell, \
                BertTrainOneStepWithLossScaleCellForAdam, \
                AdamWeightDecayForBert, AdamWeightDecayOp, \
                BertPreTraining, BertPretrainEval
from src.dataset import create_bert_dataset, create_bert_eval_dataset
from src.config import cfg, bert_net_cfg
from src.utils import LossCallBack, BertLearningRate, BertMetric
from mlperf_logging import mllog

_current_dir = os.path.dirname(os.path.realpath(__file__))

DATA_NAME = "en-wiki-20200101"
LOCAL_CACHE_PATH = "/cache_mlperf"
LOCAL_CACHE_DATA_PATH = os.path.join(LOCAL_CACHE_PATH, DATA_NAME)
os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')
job_id = os.getenv('JOB_ID')
job_id = job_id if job_id != "" else "default"
device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
rank_id = int(os.getenv('RANK_ID', '0'))

log_filename = "bert_mllog_{}.log".format(rank_id)

# Eval interval: FLOOR(0.05 * (230.23 * GBS + 3000000), 25000)
FIRST_EVAL_SAMPLES = 1000000
EVAL_INTERVAL = 500000


def argparse_init():
    """Argparse init."""
    parser = argparse.ArgumentParser(description='bert pre_training')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument("--distribute", type=str, default="false", choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--epoch_size", type=int, default="1", help="Epoch size, default is 1.")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--enable_save_ckpt", type=str, default="true", choices=["true", "false"],
                        help="Enable save checkpoint, default is true.")
    parser.add_argument("--enable_lossscale", type=str, default="true", choices=["true", "false"],
                        help="Use lossscale or not, default is not.")
    parser.add_argument("--do_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable shuffle for dataset, default is true.")
    parser.add_argument("--enable_data_sink", type=str, default="true", choices=["true", "false"],
                        help="Enable data sink, default is true.")
    parser.add_argument("--data_sink_steps", type=int, default="-1", help="Sink steps for each epoch, default is 1.")
    parser.add_argument("--accumulation_steps", type=int, default="1",
                        help="Accumulating gradients N times before weight update, default is 1.")
    parser.add_argument("--allreduce_post_accumulation", type=str, default="true", choices=["true", "false"],
                        help="Whether to allreduce after accumulation of N steps or after each step, default is true.")
    parser.add_argument("--save_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--save_checkpoint_steps", type=int, default=1000, help="Save checkpoint steps, "
                                                                                "default is 1000.")
    parser.add_argument("--train_steps", type=int, default=-1, help="Training Steps, default is -1, "
                                                                    "meaning run all steps according to epoch number.")
    parser.add_argument("--save_checkpoint_num", type=int, default=1, help="Save checkpoint numbers, default is 1.")
    parser.add_argument("--data_dir", type=str, default="", help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")
    parser.add_argument("--enable_graph_kernel", type=str, default="auto", choices=["auto", "true", "false"],
                        help="Accelerate by graph kernel, default is auto.")
    parser.add_argument("--total_steps", type=int, default=-1, help="Total steps, default is -1, "
                                                                    "meaning run all steps according to epoch number.")
    parser.add_argument("--train_with_eval", type=str, default="true", choices=['true', 'false'])
    parser.add_argument("--eval_data_dir", type=str, default="", help="Data path for evaluation")
    parser.add_argument("--data_url", type=str, default="", help="dataset url")
    parser.add_argument("--train_url", type=str, default="", help="training url")
    parser.add_argument("--enable_profile", type=ast.literal_eval, default=False, help="Enable profiling for training")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--lr_start", type=float, default=None, help="start learning rate")
    parser.add_argument("--lr_end", type=float, default=None, help="end learning rate")
    parser.add_argument("--eps", type=float, default=None, help="eps for optimizer")
    parser.add_argument("--beta1", type=float, default=None, help="Beta_1 of lamb optimizer")
    parser.add_argument("--beta2", type=float, default=None, help="Beta_2 of lamb optimizer")
    parser.add_argument("--weight_decay", type=float, default=None, help="weight of lamb optimizer")
    parser.add_argument("--warmup_steps", type=int, default=None, help="warmup steps for optimizer")
    parser.add_argument("--start_warmup_steps", type=int, default=None, help="start warmup steps for optimizer")
    parser.add_argument("--do_train", type=ast.literal_eval, default=True, help="Do training")
    parser.add_argument("--do_eval", type=ast.literal_eval, default=True, help="Do evaluation before training")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size of network")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Evaluation batch size of network")

    args_opt = parser.parse_args()

    if args_opt.batch_size is not None:
        cfg.batch_size = args_opt.batch_size
    if args_opt.lr_start != None:
        cfg.Lamb.learning_rate = args_opt.lr_start
    if args_opt.lr_end != None:
        cfg.Lamb.end_learning_rate = args_opt.lr_end
    if args_opt.eps != None:
        cfg.Lamb.eps = args_opt.eps
    if args_opt.warmup_steps != None:
        cfg.Lamb.warmup_steps = args_opt.warmup_steps
    if args_opt.start_warmup_steps != None:
        cfg.Lamb.start_warmup_steps = args_opt.start_warmup_steps
    if args_opt.beta1 != None:
        cfg.Lamb.beta1 = args_opt.beta1
    if args_opt.beta2 != None:
        cfg.Lamb.beta2 = args_opt.beta2
    if args_opt.weight_decay != None:
        cfg.Lamb.weight_decay = args_opt.weight_decay
    if args_opt.eval_batch_size is None:
        args_opt.eval_batch_size = cfg.batch_size

    return args_opt


def check_data_exist(path):
    if not os.path.isdir(path):
        print("local cache path not exist: {}".format(path))
        return False

    dir_list = os.listdir(path)
    if "eval" not in dir_list:
        print("eval dir lost")
        return False
    if "train" not in dir_list:
        print("train dir lost")
        return False

    train_count = len(os.listdir(os.path.join(path, "train")))
    if train_count != 500:
        print("train file lost, found: {}".format(train_count))
        print("Train file found: {}".format(os.listdir(os.path.join(path, "train"))))
        return False
    eval_count = len(os.listdir(os.path.join(path, "eval")))
    if eval_count != 1:
        print("eval file lost, found: {}".format(eval_count))
        print("Eval file found: {}".format(os.listdir(os.path.join(path, "eval"))))
        return False

    return True


def sync_dataset(data_url, data_dir):
    import moxing as mox
    import time
    sync_lock = "/tmp/copy_sync.lock"
    if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
        if not os.path.exists(data_dir):
            os.system('sudo mkdir {}'.format(data_dir))
            os.system('sudo chmod -R 777 {}'.format(data_dir))
        mox.file.copy_parallel(data_url, data_dir)
        print("===finish download datasets===")
        try:
            os.mknod(sync_lock)
        except:
            pass

        print("===save flag===")
    
    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)


def moxing_barrier(train_url, key="train"):
    if not train_url:
        return
    import moxing as mox
    try_cnt = 1
    while True:
        try:
            barrier_file = "{}_{}_{}.txt".format(key, job_id, rank_id)
            barrier_file = os.path.join(train_url, key, barrier_file)
            mox.file.write(barrier_file, '{}'.format(rank_id))
            print("rank_id: {}, try_cnt={}, successful write {}".format(rank_id, try_cnt, barrier_file))
            break
        except Exception as e:
            print(e)
            print("rank_id: {}, failed {} times".format(rank_id, try_cnt))
            time.sleep(3)
            try_cnt += 1

    while rank_id == 0:
        existed = []
        all_rank_exist = True
        for rank_item in range(device_num):
            if rank_item not in existed:
                rank_fn_item = os.path.join(train_url, key, '{}_{}_{}.txt'.format(key, job_id, rank_item))
                try:
                    if not mox.file.exists(rank_fn_item):
                        print("rank_fn_item:{} is not exist".format(rank_fn_item))
                        all_rank_exist = False
                        break
                    else:
                        existed.append(rank_item)
                except:
                    all_rank_exist = False
        if all_rank_exist:
            break
        else:
            time.sleep(3)
    print("Reach Barrier at time: ", time.time(), flush=True)

def moxing_copy(train_url, train_dir = "/cache/train"):
    if not train_url:
        print("train url is empty")
        return
    import moxing as mox
    try_cnt = 1
    print("Start to copy train directory")
    rank_fn_item = os.path.join(train_url, '{}_{}_{}.txt'.format("start_copy", job_id, 0))
    while True:
        if rank_id == 0:
            try:
                mox.file.copy_parallel(train_dir, train_url)
                mox.file.write(rank_fn_item, '{}'.format(rank_id))
                break
            except Exception as e:
                print(e)
                print("rank_id: {}, failed {} times".format(rank_id, try_cnt))
                try_cnt += 1
        else:
            if not mox.file.exists(rank_fn_item):
                time.sleep(1)
            else:
                break
    print("finish copy train directory ", time.time(), flush=True)

def moxing_wrapper(run_func):
    def func():
        args_opt = argparse_init()
        set_seed(args_opt.seed)
        data_dir = "/cache/data"
        train_dir = os.getcwd() 
        if os.path.isdir(LOCAL_CACHE_PATH):
            data_dir = LOCAL_CACHE_DATA_PATH
        if args_opt.data_url:
            train_dir = "/cache/train"
            if check_data_exist(data_dir):
                print("Dataset cache found: ", os.listdir(data_dir))
            else:
                sync_dataset(args_opt.data_url, data_dir)
                print(args_opt.data_url)
                print("Finish download dataset: ", os.listdir(data_dir))
            if not os.path.isdir(train_dir):
                try:
                    os.mkdir(train_dir)
                except:
                    pass
            args_opt.data_dir = os.path.join(data_dir, "train")
            args_opt.eval_data_dir = os.path.join(data_dir, "eval")
            args_opt.save_checkpoint_path = train_dir
            args_opt.device_num = device_num
            args_opt.device_id = device_id

        moxing_barrier(args_opt.train_url, "dataset")

        global log_filename
        log_filename = os.path.join(train_dir, log_filename)

        if args_opt.enable_profile:
            profiler = Profiler(output_path=train_dir)

        run_func(args_opt)

        if args_opt.enable_profile:
            profiler.analyse()

        if args_opt.train_url:
            import moxing as mox
            print("Start to copy train directory")
            if rank_id == 0:
                mox.file.copy_parallel(train_dir, args_opt.train_url)
        moxing_barrier(args_opt.train_url, "finish")
    return func 


class EvalCallback(Callback):
    def __init__(self, model, eval_ds, global_batch, mllogger, train_url=""):
        super(EvalCallback, self).__init__()
        self.model = model
        self.eval_ds = eval_ds
        self.global_batch = global_batch
        self.eval_count = 0
        self.num_samples = 1
        self.mllogger = mllogger
        self.train_url = train_url

    def epoch_begin(self, run_context):
        self.mllogger.start(key=mllog.constants.BLOCK_START, metadata={'first_epoch_num': self.num_samples,
                                                                       'epoch_count': self.num_samples})

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        num_samples = cb_params.cur_step_num * self.global_batch
        self.num_samples = num_samples
        cur_eval = num_samples // EVAL_INTERVAL
        print("num_samples: ", num_samples, " cur_eval: ", cur_eval, " eval_count: ", self.eval_count,
              " at time: ", time.time(), flush=True)
        if cur_eval > self.eval_count:
            res = self.model.eval(self.eval_ds, dataset_sink_mode=True)
            print("===========================")
            print("Accuracy is: ", "%.4f" % res, " at time: ", time.time(), " num samples: ", num_samples)
            self.mllogger.end(key=mllog.constants.BLOCK_STOP, metadata={'first_epoch_num': self.num_samples,
                                                                        'epoch_num': self.num_samples})
            self.mllogger.event(key=mllog.constants.EVAL_ACCURACY, value=res,
                                metadata={"train_samples": num_samples,
                                          "epoch_num": self.num_samples})

            print("===========================")
            if res > 0.72:
                self.mllogger.event(key=mllog.constants.RUN_STOP, metadata={"status": "success"})
                self.mllogger.event(key="train_samples", value=num_samples)
                self.mllogger.event(key="eval_samples", value=10000)
                if self.train_url:
                    moxing_copy(train_url=self.train_url)
                run_context.request_stop()
            self.eval_count += 1


def _set_bert_all_reduce_split():
    """set bert all_reduce fusion split, support num_hidden_layers is 12 and 24."""
    device_target = context.get_context('device_target')
    enable_graph_kernel = context.get_context('enable_graph_kernel')
    device_num = context.get_auto_parallel_context('device_num')
    if bert_net_cfg.num_hidden_layers == 12:
        if bert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 87, 116, 145, 174, 203, 217])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[28, 55, 82, 109, 136, 163, 190, 205])
            if device_target == 'GPU' and enable_graph_kernel and device_num == 8:
                context.set_auto_parallel_context(all_reduce_fusion_config=[180, 205])
            elif device_target == 'GPU' and enable_graph_kernel and device_num == 16:
                context.set_auto_parallel_context(all_reduce_fusion_config=[120, 205])
    elif bert_net_cfg.num_hidden_layers == 24:
        if bert_net_cfg.use_relative_positions:
            context.set_auto_parallel_context(all_reduce_fusion_config=[30, 90, 150, 210, 270, 330, 390, 421])
        else:
            context.set_auto_parallel_context(all_reduce_fusion_config=[38, 93, 148, 203, 258, 313, 368, 397])


def _get_optimizer(args_opt, network):
    """get bert optimizer, support Lamb, Momentum, AdamWeightDecay."""
    if cfg.optimizer != 'Lamb':
        raise ValueError("Only support Lamb for mlperf")

    lr_schedule = BertLearningRate(learning_rate=cfg.Lamb.learning_rate,
                                    end_learning_rate=cfg.Lamb.end_learning_rate,
                                    warmup_steps=cfg.Lamb.warmup_steps,
                                    start_warmup_steps=cfg.Lamb.start_warmup_steps,
                                    decay_steps=args_opt.total_steps,
                                    power=cfg.Lamb.power)
    params = network.trainable_params()
    decay_params = list(filter(cfg.Lamb.decay_filter, params))
    other_params = list(filter(lambda x: not cfg.Lamb.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': cfg.Lamb.weight_decay},
                    {'params': other_params},
                    {'order_params': params}]
    optimizer = Lamb(group_params, learning_rate=lr_schedule, eps=cfg.Lamb.eps,
                        beta1=cfg.Lamb.beta1, beta2=cfg.Lamb.beta2)
    return optimizer


def _auto_enable_graph_kernel(device_target, graph_kernel_mode):
    """Judge whether is suitable to enable graph kernel."""
    return graph_kernel_mode in ("auto", "true") and device_target == 'GPU' and \
        cfg.bert_network == 'base' and cfg.optimizer == 'AdamWeightDecay'


def _set_graph_kernel_context(device_target, enable_graph_kernel, is_auto_enable_graph_kernel):
    if enable_graph_kernel == "true" or is_auto_enable_graph_kernel:
        if device_target == 'GPU':
            context.set_context(enable_graph_kernel=True)
        else:
            logger.warning('Graph kernel only supports GPU back-end now, run with graph kernel off.')


def _check_compute_type(args_opt, is_auto_enable_graph_kernel):
    if args_opt.device_target == 'GPU' and bert_net_cfg.compute_type != mstype.float32 and \
       not is_auto_enable_graph_kernel:
        warning_message = 'Gpu only support fp32 temporarily, run with fp32.'
        bert_net_cfg.compute_type = mstype.float32
        if args_opt.enable_lossscale == "true":
            args_opt.enable_lossscale = "false"
            warning_message = 'Gpu only support fp32 temporarily, run with fp32 and disable lossscale.'
        logger.warning(warning_message)


@moxing_wrapper
def run_pretrain(args_opt):
    """pre-train bert_clue"""
    print("config: ", cfg)
    print("args:", args_opt)

    print("mllog file: ", log_filename)
    mllog.config(filename=log_filename)
    mllog.config(
        default_namespace="mindspore",
        default_stack_offset=1,
        default_clear_line=False,
        root_dir=os.path.normpath(
            os.path.dirname(os.path.realpath(__file__))))
    mllogger = mllog.get_mllogger()
    mllogger.event(key=mllog.constants.SUBMISSION_BENCHMARK, value="bert")
    mllogger.event(key=mllog.constants.SUBMISSION_DIVISION, value="closed")
    mllogger.event(key=mllog.constants.SUBMISSION_ORG, value="PCL")
    mllogger.event(key=mllog.constants.SUBMISSION_PLATFORM, value="Ascend 910A")
    mllogger.event(key=mllog.constants.SUBMISSION_STATUS, value="research")
    mllogger.event(key=mllog.constants.CACHE_CLEAR)

    global_batchsize = args_opt.device_num * cfg.batch_size
    # Eval interval: FLOOR(0.05 * (230.23 * GBS + 3000000), 25000)
    global EVAL_INTERVAL
    EVAL_INTERVAL = math.floor(0.05 * (230.23 * global_batchsize + 3000000) / 25000)
    EVAL_INTERVAL *= 25000
    print("EVAL_INTERVAL: ", EVAL_INTERVAL)
    start_time = time.time()
    print("start time: ", start_time)
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
    context.set_context(reserve_class_name_in_scope=False)
    is_auto_enable_graph_kernel = _auto_enable_graph_kernel(args_opt.device_target, args_opt.enable_graph_kernel)
    _set_graph_kernel_context(args_opt.device_target, args_opt.enable_graph_kernel, is_auto_enable_graph_kernel)
    ckpt_save_dir = args_opt.save_checkpoint_path
    if args_opt.distribute == "true":
        if args_opt.device_target == 'Ascend':
            D.init()
            device_num = args_opt.device_num
            rank = D.get_rank()
            print("Device_num: ", device_num)
            print("rank_id: ", rank_id)
            print("rank: ", rank)
            print("Group_size: ", D.get_group_size())
        else:
            D.init()
            device_num = D.get_group_size()
            rank = D.get_rank()
        ckpt_save_dir = args_opt.save_checkpoint_path + 'ckpt_' + str(get_rank()) + '/'

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        _set_bert_all_reduce_split()
    else:
        rank = 0
        device_num = 1

    _check_compute_type(args_opt, is_auto_enable_graph_kernel)

    if args_opt.data_sink_steps == -1:
        args_opt.data_sink_steps = math.ceil(EVAL_INTERVAL / (device_num * cfg.batch_size * args_opt.accumulation_steps))

    if args_opt.accumulation_steps > 1:
        logger.info("accumulation steps: {}".format(args_opt.accumulation_steps))
        logger.info("global batch size: {}".format(cfg.batch_size * args_opt.accumulation_steps))
        if args_opt.enable_data_sink == "true":
            args_opt.data_sink_steps *= args_opt.accumulation_steps
            logger.info("data sink steps: {}".format(args_opt.data_sink_steps))
        if args_opt.enable_save_ckpt == "true":
            args_opt.save_checkpoint_steps *= args_opt.accumulation_steps
            logger.info("save checkpoint steps: {}".format(args_opt.save_checkpoint_steps))

    ds = create_bert_dataset(device_num, rank, args_opt.do_shuffle, args_opt.data_dir, args_opt.schema_dir)
    print("create dataset time: ", time.time() - start_time, flush=True);

    bert = BertPreTraining(bert_net_cfg, True, False)
    net_with_loss = BertNetworkWithLoss(bert_net_cfg, True, bert=bert)

    print("net_with_loss time: ", time.time() - start_time, flush=True);

    new_repeat_count = args_opt.epoch_size * ds.get_dataset_size() // args_opt.data_sink_steps
    if args_opt.train_steps > 0:
        train_steps = args_opt.train_steps * args_opt.accumulation_steps
        new_repeat_count = min(new_repeat_count, train_steps // args_opt.data_sink_steps)
    else:
        args_opt.train_steps = args_opt.epoch_size * ds.get_dataset_size() // args_opt.accumulation_steps
        logger.info("train steps: {}".format(args_opt.train_steps))

    args_opt.total_steps = args_opt.train_steps if args_opt.total_steps == -1 else args_opt.total_steps

    optimizer = _get_optimizer(args_opt, net_with_loss)
    callback = [TimeMonitor(args_opt.data_sink_steps), LossCallBack(ds.get_dataset_size())]
    if args_opt.enable_save_ckpt == "true" and args_opt.device_id % min(8, device_num) == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args_opt.save_checkpoint_steps,
                                     keep_checkpoint_max=args_opt.save_checkpoint_num)
        ckpoint_cb = ModelCheckpoint(prefix='checkpoint_bert',
                                     directory=None if ckpt_save_dir == "" else ckpt_save_dir, config=config_ck)
        callback.append(ckpoint_cb)

    if args_opt.load_checkpoint_path:
        param_dict = load_checkpoint(args_opt.load_checkpoint_path)
        net_param = net_with_loss.parameters_dict()
        load_param_into_net(net_with_loss, param_dict)

    if args_opt.enable_lossscale == "true":
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale_value,
                                                 scale_factor=cfg.scale_factor,
                                                 scale_window=cfg.scale_window)
        accumulation_steps = args_opt.accumulation_steps
        enable_global_norm = cfg.enable_global_norm
        if accumulation_steps <= 1:
            if cfg.optimizer == 'AdamWeightDecay' and args_opt.device_target == 'GPU':
                net_with_grads = BertTrainOneStepWithLossScaleCellForAdam(net_with_loss, optimizer=optimizer,
                                                                          scale_update_cell=update_cell)
            else:
                net_with_grads = BertTrainOneStepWithLossScaleCell(net_with_loss, optimizer=optimizer,
                                                                   scale_update_cell=update_cell)
        else:
            allreduce_post = args_opt.distribute == "false" or args_opt.allreduce_post_accumulation == "true"
            net_with_accumulation = (BertTrainAccumulationAllReducePostWithLossScaleCell if allreduce_post else
                                     BertTrainAccumulationAllReduceEachWithLossScaleCell)
            net_with_grads = net_with_accumulation(net_with_loss, optimizer=optimizer,
                                                   scale_update_cell=update_cell,
                                                   accumulation_steps=accumulation_steps,
                                                   enable_global_norm=enable_global_norm)
    else:
        net_with_grads = BertTrainOneStepCell(net_with_loss, optimizer=optimizer)

    print("net_with_grads time: ", time.time() - start_time, flush=True);

    mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=cfg.batch_size * device_num)
    mllogger.event(key="gradient_accumulation_steps", value=args_opt.accumulation_steps)
    mllogger.event(key="seed", value=args_opt.seed)
    mllogger.event(key="opt_name", value="lamb")
    mllogger.event(key=mllog.constants.OPT_BASE_LR, value=cfg.Lamb.learning_rate)
    mllogger.event(key=mllog.constants.OPT_LAMB_LR_MIN, value=cfg.Lamb.end_learning_rate)
    mllogger.event(key=mllog.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=cfg.Lamb.power)
    mllogger.event(key="opt_learning_rate_training_steps", value=args_opt.total_steps)
    mllogger.event(key="opt_learning_rate_warmup_steps", value=cfg.Lamb.warmup_steps)
    mllogger.event(key="num_warmup_steps", value=cfg.Lamb.warmup_steps)
    mllogger.event(key="opt_epsilon", value=cfg.Lamb.eps)
    mllogger.event(key="opt_lamb_beta_1", value=cfg.Lamb.beta1)
    mllogger.event(key="opt_lamb_beta_2", value=cfg.Lamb.beta2)
    mllogger.event(key="opt_lamb_weight_decay_rate", value=cfg.Lamb.weight_decay)
    mllogger.event(key="start_warmup_step", value=cfg.Lamb.start_warmup_steps)

    mllogger.start(key=mllog.constants.INIT_START)

    eval_ds = create_bert_eval_dataset(args_opt.eval_batch_size, device_num, rank, args_opt.eval_data_dir, None)
    net_eval = BertPretrainEval(bert_net_cfg, bert=bert)
    print("eval phase: ", net_eval.phase)
    model = Model(net_with_grads, eval_network=net_eval, metrics={'bert_acc': BertMetric(cfg.batch_size)})
    model._init(ds, None, args_opt.data_sink_steps, new_repeat_count)

    if args_opt.do_eval:
        res = model.eval(eval_ds, dataset_sink_mode=True)
        print("===========================")
        print("Accuracy is: ", "%.4f" % res, " at time: ", time.time())
        print("===========================")
        mllogger.event(key=mllog.constants.EVAL_ACCURACY, value=res,
                            metadata={"train_samples": 0,
                                      "epoch_num": 0})

    if args_opt.train_with_eval == 'true':
        eval_callback = EvalCallback(model, eval_ds, device_num * cfg.batch_size, mllogger, args_opt.train_url)
        callback.append(eval_callback)
        model._init(None, eval_ds, args_opt.data_sink_steps, new_repeat_count)

    print("initialization time: ", time.time() - start_time, " at time: ", time.time(), flush=True);

    if args_opt.train_url:
        moxing_barrier(args_opt.train_url)
    else:
        if rank == 0:
            time.sleep(100)
    mllogger.end(key=mllog.constants.INIT_STOP)

    start_time = time.time()
    print("start running time: ", start_time, flush=True)

    mllogger.start(key=mllog.constants.RUN_START)

    if args_opt.do_train:
        model.train(new_repeat_count, ds, callbacks=callback,
                    dataset_sink_mode=(args_opt.enable_data_sink == "true"), sink_size=args_opt.data_sink_steps)
    end_time = time.time()
    print("finish time: ", end_time, ", time cost: ", end_time - start_time)


if __name__ == '__main__':
    print("current path: ", os.getcwd())
    run_pretrain()

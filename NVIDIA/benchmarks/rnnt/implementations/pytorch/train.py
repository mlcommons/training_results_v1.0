# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import random
import time

import torch
import multiprocessing
import numpy as np
import torch.distributed as dist
from apex import amp
from torch.cuda.amp import GradScaler
from apex.optimizers import FusedLAMB
from apex.parallel import DistributedDataParallel
from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
import amp_C
import math

from common import helpers
from common.data.dali import sampler as dali_sampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.text import Tokenizer
from common.data import features
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch, Preproc)
from common.optimizers import lr_policy
from common.tb_dllogger import flush_log, init_log, log
from rnnt import config
from rnnt.decoder import RNNTGreedyDecoder
from rnnt.loss import RNNTLoss
from rnnt.loss import apexTransducerLoss
from rnnt.model import RNNT
from rnnt.rnnt_graph import RNNTGraph

from mlperf import logging


# TODO Eval batch size

def parse_args():
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', default=100, type=int,
                          help='Number of epochs for the entire training')
    training.add_argument("--warmup_epochs", default=6, type=int,
                          help='Initial epochs of increasing learning rate')
    training.add_argument("--hold_epochs", default=40, type=int,
                          help='Constant max learning rate epochs after warmup')
    training.add_argument('--epochs_this_job', default=0, type=int,
                          help=('Run for a number of epochs with no effect on the lr schedule.'
                                'Useful for re-starting the training.'))
    training.add_argument('--cudnn_benchmark', action='store_true', default=True,
                          help='Enable cudnn benchmark')
    training.add_argument('--amp_level', default=1, type=int, choices=[0, 1, 2, 3],
                          help='APEX AMP optimization level')
    training.add_argument('--seed', default=None, type=int, help='Random seed')
    training.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                          help='GPU id used for distributed training')
    training.add_argument('--target', default=0.058, type=float, help='Target WER accuracy')
    training.add_argument('--apex_transducer_loss', default=None, type=str, choices=['fp16', 'fp32'], 
                            help='what precision of apex transducer_loss to use')
    training.add_argument('--fuse_relu_dropout', action='store_true', 
                            help='Fuse ReLU and dropout in the joint net')
    training.add_argument('--weights_init_scale', default=0.5, type=float, help='If set, overwrites value in config.')
    training.add_argument('--hidden_hidden_bias_scale', type=float, help='If set, overwrites value in config.')
    training.add_argument('--batch_eval_mode', default=None, type=str, choices=['no_cg', 'cg', 'cg_unroll_pipeline'],
                    help='do evaluation in batch')
    training.add_argument('--cg_unroll_factor', default=4, type=int, help='Unrolling factor for batch eval mode cg_unroll_pipeline')
    training.add_argument('--apex_transducer_joint', default=None, type=str, choices=['pack', 'not_pack'], 
                            help='whether or not to pack the sequence with apex transducer_joint')
    training.add_argument('--buffer_pre_alloc', action='store_true', 
                            help='Pre-allocate buffer in PyTorch')
    training.add_argument('--multilayer_lstm', action='store_true', 
                            help='Use multilayer LSTMs instead of splitting them into multiple single-layer ones')
    training.add_argument('--batch_split_factor', default=1, type=int, help='Split batches feed into the joint net')
    training.add_argument('--apex_mlp', action='store_true', 
                            help='Use apex MLP')

    training.add_argument("--num_cg", default=0, type=int,
                          help='number of graphs needed for training')
    training.add_argument('--min_seq_split_len', default=-1, type=int, help='Split sequences in a mini-batch to improve performance')
    training.add_argument('--pre_sort_for_seq_split', action='store_true', 
                            help='Presort samples in a mini-batch so that seq split is more effective')


    optim = parser.add_argument_group('optimization setup')
    optim.add_argument('--batch_size', default=128, type=int,
                       help='Effective batch size per GPU (might require grad accumulation')
    optim.add_argument('--val_batch_size', default=2, type=int,
                       help='Evalution time batch size')
    optim.add_argument('--lr', default=4e-3, type=float,
                       help='Peak learning rate')
    optim.add_argument("--min_lr", default=1e-5, type=float,
                       help='minimum learning rate')
    optim.add_argument("--lr_exp_gamma", default=0.935, type=float,
                       help='gamma factor for exponential lr scheduler')
    optim.add_argument('--weight_decay', default=1e-3, type=float,
                       help='Weight decay for the optimizer')
    optim.add_argument('--grad_accumulation_steps', default=8, type=int,
                       help='Number of accumulation steps')
    optim.add_argument('--clip_norm', default=1, type=float,
                       help='If provided, gradients will be clipped above this norm')
    optim.add_argument('--beta1', default=0.9, type=float, help='Beta 1 for optimizer')
    optim.add_argument('--beta2', default=0.999, type=float, help='Beta 2 for optimizer')
    optim.add_argument('--ema', type=float, default=0.999,
                       help='Discount factor for exp averaging of model weights')
    optim.add_argument('--multi_tensor_ema', action='store_true', 
                            help='Use multi_tensor_apply for EMA')
    optim.add_argument('--dist_lamb', action='store_true', 
                            help='Use distributed LAMB')
    optim.add_argument('--ema_update_type', default='fp32', type=str, choices=['fp16', 'fp32'], 
                            help='is ema applied on the fp32 master weight or fp16 weight')
    optim.add_argument('--dwu_group_size', default=8, type=int,
                       help='Group size for distributed optimizer. Will be ignored if non-distributed optimizer is used')



    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--dali_device', type=str, choices=['cpu', 'gpu'],
                    default='cpu', help='Use DALI pipeline for fast data processing')
    io.add_argument('--resume', action='store_true',
                    help='Try to resume from last saved checkpoint.')
    io.add_argument('--ckpt', default=None, type=str,
                    help='Path to a checkpoint for resuming training')
    io.add_argument('--save_at_the_end', action='store_true',
                    help='Saves model checkpoint at the end of training')
    io.add_argument('--save_frequency', default=None, type=int,
                    help='Checkpoint saving frequency in epochs')
    io.add_argument('--keep_milestones', default=[], type=int, nargs='+',
                    help='Milestone checkpoints to keep from removing')
    io.add_argument('--save_best_from', default=200, type=int,
                    help='Epoch on which to begin tracking best checkpoint (dev WER)')
    io.add_argument('--val_frequency', default=1, type=int,
                    help='Number of epochs between evaluations on dev set')
    io.add_argument('--log_frequency', default=25, type=int,
                    help='Number of steps between printing training stats')
    io.add_argument('--prediction_frequency', default=None, type=int,
                    help='Number of steps between printing sample decodings')
    io.add_argument('--model_config', default='configs/baseline_v3-1023sp.yaml',
                    type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--num_buckets', type=int, default=6,
                    help='If provided, samples will be grouped by audio duration, '
                         'to this number of backets, for each bucket, '
                         'random samples are batched, and finally '
                         'all batches are randomly shuffled')
    io.add_argument('--vectorized_sampler', action='store_true',
                    help='Use optimized bucketing sampler implementation')
    io.add_argument('--dist_sampler', action='store_true',
                    help='Each rank owns an unique copy of file list')
    io.add_argument('--train_manifests', type=str, required=True, nargs='+',
                    help='Paths of the training dataset manifest file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
                    help='Paths of the evaluation datasets manifest files')
    io.add_argument('--max_duration', type=float,
                    help='Discard samples longer than max_duration')
    io.add_argument('--max_txt_len', type=int, default=125,
                    help='The longest text length in the sample')
    io.add_argument('--max_eval_sample_duration', type=float, default=32.7,
                    help='The max duration of samples in the eval set')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    io.add_argument('--output_dir', type=str, required=True,
                    help='Directory for logs and checkpoints')
    io.add_argument('--log_file', type=str, default=None,
                    help='Path to save the training logfile.')
    io.add_argument('--max_symbol_per_sample', type=int, default=None,
                    help='maximum number of symbols per sample can have during eval')
    io.add_argument('--data_cpu_threads', type=int, default=multiprocessing.cpu_count(),
            help='Number of CPU threads used for data loading and preprocessing.')
    io.add_argument('--synthetic_audio_seq_len', type=int, default=None,
                    help='length for synthetic audio sequence.')
    io.add_argument('--synthetic_text_seq_len', type=int, default=None,
                    help='length for synthetic text sequence.')
    io.add_argument('--enable_seq_len_stats', action='store_true', 
                            help='Store and output seq len stats')
    io.add_argument('--vectorized_sa', action='store_true',
                    help='Vectorized implementation of SpecAugment')
    io.add_argument('--in_mem_file_list', action='store_true',
                    help='prepare file list in memory instead of on the disk')
    io.add_argument('--enable_prefetch', action='store_true',
                    help='prefetch and preprocess input data for next iteration')
    io.add_argument('--tokenized_transcript', action='store_true',
                    help='loads transcript in tokenized form')
    io.add_argument('--jit_tensor_formation', action='store_true',
                    help='just-in-time tensor formation. Form the input txt tensor on the fly.')
    io.add_argument('--dali_dont_use_mmap', action='store_true',
                    help='Disable mmap for DALI')
    return parser.parse_args()


@torch.no_grad()
def apply_ema(optimizer, model, ema_model, decay, ema_update_type):
    if not decay:
        return

    ema_model_weight_list = list(ema_model.parameters())
    if ema_update_type == "fp32":
        model_weight_list = list(amp.master_params(optimizer))
    elif ema_update_type == "fp16":
        model_weight_list = list(model.parameters())
    for ema, source in zip(ema_model_weight_list, model_weight_list):
        ema.copy_(decay * ema + (1 - decay) * source)

def init_multi_tensor_ema(optimizer, model, ema_model, ema_update_type):
    # Regardless of the ema_update_type, ema_accumulation_type is FP32, which is
    # ensured by FP32 ema_model
    ema_model_weight_list = list(ema_model.parameters())
    if ema_update_type == "fp32":
        model_weight_list = list(amp.master_params(optimizer))
    elif ema_update_type == "fp16":
        model_weight_list = list(model.parameters())
    overflow_buf_for_ema = torch.cuda.IntTensor([0])
    return ema_model_weight_list, model_weight_list, overflow_buf_for_ema

def apply_multi_tensor_ema(model_weight_list, ema_model_weight_list, decay, overflow_buf):
    if not decay:
        return

    amp_C.multi_tensor_axpby(65536, overflow_buf, [ema_model_weight_list, model_weight_list, ema_model_weight_list], decay, 1-decay, -1)




@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, detokenize,
             ema_model, loss_fn, greedy_decoder, amp_level):
    logging.log_start(logging.constants.EVAL_START, metadata=dict(epoch_num=epoch))
    start_time = time.time()
    agg = {'preds': [], 'txts': [], 'idx': []}
    greedy_decoder.update_ema_model_eval(ema_model)

    for i, batch in enumerate(val_loader):
        # print(f'{val_loader.pipeline_type} evaluation: {i:>10}/{len(val_loader):<10}', end='\r')

        audio, audio_lens, txt, txt_lens = batch

        feats, feat_lens = val_feat_proc([audio, audio_lens])
        if amp_level == 2:
            feats = feats.half()
        
        pred = greedy_decoder.decode(feats, feat_lens)
        agg['preds'] += helpers.gather_predictions([pred], detokenize)
        agg['txts'] += helpers.gather_transcripts([txt.cpu()], [txt_lens.cpu()], detokenize)

    wer, loss = process_evaluation_epoch(agg)

    logging.log_event(logging.constants.EVAL_ACCURACY, value=wer, metadata=dict(epoch_num=epoch))
    logging.log_end(logging.constants.EVAL_STOP, metadata=dict(epoch_num=epoch))

    log((epoch,), step, 'dev_ema', {'wer': 100.0 * wer,
                                 'took': time.time() - start_time})
    return wer


def train_step( model, loss_fn, args, batch_size, feats, feat_lens, txt, txt_lens, optimizer, grad_scaler, 
                meta_data, train_loader, rnnt_graph, copy_stream, pred_stream):
    # sync free loss
    lr_cpu = torch.tensor(0, dtype=torch.float, device='cpu').pin_memory()
    loss_cpu = torch.tensor(0, dtype=torch.float16, device='cpu').pin_memory()
    if args.batch_split_factor == 1:
        if rnnt_graph is not None:
            log_probs, log_prob_lens = rnnt_graph.step(feats, feat_lens, txt, txt_lens, meta_data[0])
        else:    
            log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens, meta_data[0])

        loss = loss_fn(log_probs, log_prob_lens, txt, txt_lens, meta_data[0])
        if args.enable_prefetch and train_loader is not None:
            # if train_loader is None, that means we are doing dummy runs,
            # so we don't need to prefetch
            train_loader.data_iterator().prefetch()
        loss /= args.grad_accumulation_steps

        # copy loss to cpu asynchronously
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            loss_cpu.copy_(loss.detach(), non_blocking=True)
            if args.dist_lamb:
                lr_cpu.copy_(optimizer._lr, non_blocking=True)

        del log_probs, log_prob_lens

        if args.dist_lamb:
            grad_scaler.scale(loss).backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # sync before return         
        copy_stream.synchronize()
        if torch.isnan(loss_cpu).any():
            raise Exception("Loss is NaN")

        return loss_cpu.item(), lr_cpu.item()

    else:
        f, g, log_prob_lens = model.enc_pred(feats, feat_lens, txt, txt_lens, pred_stream)
        f_2, g_2 = f.detach(), g.detach()
        f_2.requires_grad = True
        g_2.requires_grad = True
        B_split = batch_size // args.batch_split_factor
        loss_item = 0
        for i in range(args.batch_split_factor):
            
            log_probs = model.joint(f_2[i*B_split:(i+1)*B_split], g_2[i*B_split:(i+1)*B_split], args.apex_transducer_joint, 
                                    log_prob_lens[i*B_split:(i+1)*B_split], meta_data[i])
            loss = loss_fn( log_probs, log_prob_lens[i*B_split:(i+1)*B_split], txt[i*B_split:(i+1)*B_split], 
                            txt_lens[i*B_split:(i+1)*B_split], meta_data[i])

            if args.enable_prefetch and train_loader is not None and i == 0:
                # if train_loader is None, that means we are doing dummy runs,
                # so we don't need to prefetch
                train_loader.data_iterator().prefetch()
            
            loss /= (args.grad_accumulation_steps*args.batch_split_factor)

            # copy loss to cpu asynchronously
            copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                loss_cpu.copy_(loss.detach(), non_blocking=True)
                if args.dist_lamb and i == 0:
                    # only need to copy lr for once
                    lr_cpu.copy_(optimizer._lr, non_blocking=True)

            del log_probs

            if args.dist_lamb:
                grad_scaler.scale(loss).backward()
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

            copy_stream.synchronize()
            if torch.isnan(loss_cpu).any():
                raise Exception("Loss is NaN")

            loss_item += loss_cpu.item()
        
        f.backward(f_2.grad)
        g.backward(g_2.grad)

        return loss_item, lr_cpu.item()

def main():
    args = parse_args()
    logging.configure_logger(args.output_dir, 'RNNT')
    logging.log_start(logging.constants.INIT_START)

    assert(torch.cuda.is_available())
    assert args.prediction_frequency is None or args.prediction_frequency % args.log_frequency == 0

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # set up distributed training
    multi_gpu = args.dist_lamb or (int(os.environ.get('WORLD_SIZE', 1)) > 1)
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        print_once(f'Distributed training with {world_size} GPUs\n')
    else:
        world_size = 1

    if args.seed is not None:
        logging.log_event(logging.constants.SEED, value=args.seed)
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)
        # np_rng is used for buckets generation, and needs the same seed on every worker
        sampler_seed = args.seed
        if multi_gpu:
            dali_seed = args.seed + dist.get_rank()
        else:
            dali_seed = args.seed

    init_log(args)
    cfg = config.load(args.model_config)
    config.apply_duration_flags(cfg, args.max_duration)

    assert args.grad_accumulation_steps >= 1
    assert args.batch_size % args.grad_accumulation_steps == 0, f'{args.batch_size} % {args.grad_accumulation_steps} != 0'
    logging.log_event(logging.constants.GRADIENT_ACCUMULATION_STEPS, value=args.grad_accumulation_steps)
    batch_size = args.batch_size // args.grad_accumulation_steps
    if args.batch_split_factor != 1:
        assert batch_size % args.batch_split_factor == 0, f'{batch_size} % {args.batch_split_factor} != 0'
        assert args.dist_lamb, "dist LAMB must be used when batch split is enabled"

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)
    logging.log_event(logging.constants.SUBMISSION_BENCHMARK, value=logging.constants.RNNT)
    logging.log_event(logging.constants.SUBMISSION_ORG, value='NVIDIA')
    logging.log_event(logging.constants.SUBMISSION_DIVISION, value=logging.constants.CLOSED) # closed or open
    logging.log_event(logging.constants.SUBMISSION_STATUS, value=logging.constants.ONPREM) # on-prem/cloud/research
    logging.log_event(logging.constants.SUBMISSION_PLATFORM, value=f'{num_nodes}xSUBMISSION_PLATFORM_PLACEHOLDER')

    # set up the model
    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

    rnnt_config = config.rnnt(cfg)
    logging.log_event(logging.constants.MODEL_WEIGHTS_INITIALIZATION_SCALE, value=args.weights_init_scale)
    if args.fuse_relu_dropout:
        rnnt_config["fuse_relu_dropout"] = True
    if args.apex_transducer_joint is not None:
        rnnt_config["apex_transducer_joint"] = args.apex_transducer_joint
    if args.weights_init_scale is not None:
        rnnt_config['weights_init_scale'] = args.weights_init_scale
    if args.hidden_hidden_bias_scale is not None:
        rnnt_config['hidden_hidden_bias_scale'] = args.hidden_hidden_bias_scale
    if args.multilayer_lstm:
        rnnt_config["decoupled_rnns"] = False
    if args.apex_mlp:
        rnnt_config["apex_mlp"] = True
    enc_stack_time_factor = rnnt_config["enc_stack_time_factor"]
    model = RNNT(n_classes=tokenizer.num_labels + 1, **rnnt_config)
    model.cuda()
    blank_idx = tokenizer.num_labels
    if args.apex_transducer_loss == None:
        loss_fn = RNNTLoss(blank_idx=blank_idx)
    else:
        if args.apex_transducer_loss == "fp16":
            assert args.amp_level in [1, 2], "model in FP32 and loss in FP16 is not a valid use case"
        loss_fn = apexTransducerLoss(blank_idx, args.apex_transducer_loss, packed_input=args.apex_transducer_joint=="pack")

    # set up evaluation
    logging.log_event(logging.constants.EVAL_MAX_PREDICTION_SYMBOLS, value=args.max_symbol_per_sample)
    greedy_decoder = RNNTGreedyDecoder( blank_idx=blank_idx,
                                        batch_eval_mode=args.batch_eval_mode,
                                        cg_unroll_factor = args.cg_unroll_factor,
                                        rnnt_config=rnnt_config,
                                        max_symbol_per_sample=args.max_symbol_per_sample,
                                        amp_level=args.amp_level)

    print_once(f'Model size: {num_weights(model) / 10**6:.1f}M params\n')

    if args.ema > 0:
        ema_model = copy.deepcopy(model).cuda()
    else:
        ema_model = None
    logging.log_event(logging.constants.MODEL_EVAL_EMA_FACTOR, value=args.ema)
    
    # set up optimization
    opt_eps=1e-9
    if args.dist_lamb:
        model.half()
        initial_lrs = args.lr * torch.tensor(1, device='cuda', dtype=torch.float)
        kw = {  'params': model.parameters(), 
                'lr': initial_lrs,
                'weight_decay': args.weight_decay,
                'eps': opt_eps,
                'betas': (args.beta1, args.beta2),
                'max_grad_norm': args.clip_norm,
                'overlap_reductions': True,
                'dwu_group_size': args.dwu_group_size,
                'use_nvlamb': False,
                "dwu_num_blocks": 2,
                "dwu_num_chunks": 2,
                "dwu_num_rs_pg": 1,
                "dwu_num_ar_pg": 1,
                "dwu_num_ag_pg": 2,
                "bias_correction": True
                }
        optimizer = DistributedFusedLAMB(**kw)
        # Prevent following communicators to lock the tree
        os.environ["NCCL_SHARP_DISABLE"] = "1"
        grad_scaler = GradScaler(init_scale=512)
        print_once(f'Starting with LRs: {initial_lrs}')
    else:
        grad_scaler = None
        kw = {'params': model.parameters(), 'lr': args.lr,
              'max_grad_norm': args.clip_norm,
              'weight_decay': args.weight_decay}

        initial_lrs = args.lr

        print_once(f'Starting with LRs: {initial_lrs}')
        optimizer = FusedLAMB(betas=(args.beta1, args.beta2), eps=opt_eps, **kw)

        model, optimizer = amp.initialize(
            models=model,
            optimizers=optimizer,
            opt_level=f'O{args.amp_level}',
            max_loss_scale=512.0,
            cast_model_outputs=torch.float16 if args.amp_level == 2 else None)

    adjust_lr = lambda step, epoch: lr_policy(
            step, epoch, initial_lrs, optimizer, steps_per_epoch=steps_per_epoch,
            warmup_epochs=args.warmup_epochs, hold_epochs=args.hold_epochs,
            min_lr=args.min_lr, exp_gamma=args.lr_exp_gamma, dist_lamb=args.dist_lamb)

    if not args.dist_lamb and multi_gpu:
        model = DistributedDataParallel(model)

    print_once('Setting up datasets...')
    (
        train_dataset_kw,
        train_features_kw,
        train_splicing_kw,
        train_padalign_kw,
        train_specaugm_kw,
    ) = config.input(cfg, 'train')
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_padalign_kw,
        val_specaugm_kw,
    ) = config.input(cfg, 'val')

    logging.log_event(logging.constants.DATA_TRAIN_MAX_DURATION,
                      value=train_dataset_kw['max_duration'])
    logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MAX,
                      value=train_dataset_kw['speed_perturbation']['max_rate'])
    logging.log_event(logging.constants.DATA_SPEED_PERTURBATON_MIN,
                      value=train_dataset_kw['speed_perturbation']['min_rate'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_N,
                      value=train_specaugm_kw['freq_masks'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MIN,
                      value=train_specaugm_kw['min_freq'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_FREQ_MAX,
                      value=train_specaugm_kw['max_freq'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_N,
                      value=train_specaugm_kw['time_masks'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MIN,
                      value=train_specaugm_kw['min_time'])
    logging.log_event(logging.constants.DATA_SPEC_AUGMENT_TIME_MAX,
                      value=train_specaugm_kw['max_time'])
    logging.log_event(logging.constants.GLOBAL_BATCH_SIZE,
                      value=batch_size * world_size * args.grad_accumulation_steps)

    class PermuteAudio(torch.nn.Module):
        def forward(self, x):
            return (x[0].permute(2, 0, 1), *x[1:])

    if not train_specaugm_kw:
        train_specaugm = torch.nn.Identity()
    elif not args.vectorized_sa:
        train_specaugm = features.SpecAugment(optim_level=args.amp_level, **train_specaugm_kw)
    else:
        train_specaugm = features.VectorizedSpecAugment(optim_level=args.amp_level, **train_specaugm_kw)
    train_augmentations = torch.nn.Sequential(
        train_specaugm,
        features.FrameSplicing(optim_level=args.amp_level, **train_splicing_kw),
        features.FillPadding(optim_level=args.amp_level, ),
        features.PadAlign(optim_level=args.amp_level, **train_padalign_kw),
        PermuteAudio(),
    )

    if not val_specaugm_kw:
        val_specaugm = torch.nn.Identity()
    elif not args.vectorized_sa:
        val_specaugm = features.SpecAugment(optim_level=args.amp_level, **val_specaugm_kw)
    else:
        val_specaugm = features.VectorizedSpecAugment(optim_level=args.amp_level, **val_specaugm_kw)
    val_augmentations = torch.nn.Sequential(
        val_specaugm,
        features.FrameSplicing(optim_level=args.amp_level, **val_splicing_kw),
        features.FillPadding(optim_level=args.amp_level, ),
        features.PadAlign(optim_level=args.amp_level, **val_padalign_kw),
        PermuteAudio(),
    )

    train_feat_proc = train_augmentations
    val_feat_proc   = val_augmentations

    train_feat_proc.cuda()
    val_feat_proc.cuda()

    train_preproc = Preproc(train_feat_proc, args.dist_lamb, args.apex_transducer_joint, args.batch_split_factor, cfg)


    # graphing
    if args.num_cg > 0:
        if not args.dist_lamb:
            raise NotImplementedError("Currently CUDA graph training only works with dist LAMB")
        if args.batch_split_factor != 1:
            raise NotImplementedError("Currently CUDA graph training does not work with batch split")

        max_seq_len = math.ceil(train_preproc.audio_duration_to_seq_len(
                                                    cfg['input_train']['audio_dataset']['max_duration'], 
                                                    after_subsampling=True,
                                                    after_stack_time=False
                                                    ) 
                        * cfg["input_train"]["audio_dataset"]["speed_perturbation"]["max_rate"])

        print_once(f'Graph with max_seq_len of %d' % max_seq_len)
        rnnt_graph = RNNTGraph(model, rnnt_config, batch_size, max_seq_len, args.max_txt_len, args.num_cg)
        rnnt_graph.capture_graph()
    else:
        rnnt_graph = None

    # capture CG for eval
    if type(args.batch_eval_mode) == str and args.batch_eval_mode.startswith("cg"):
        max_seq_len = train_preproc.audio_duration_to_seq_len(  args.max_eval_sample_duration, 
                                                                after_subsampling=True, 
                                                                after_stack_time=True) 
        dict_meta_data = {"batch": args.val_batch_size, "max_feat_len": max_seq_len}
        greedy_decoder.capture_cg_for_eval(ema_model, dict_meta_data)

    # warm up optimizer
    def optimizer_warm_up():
        WARMUP_LEN = 8
        feats = torch.ones(WARMUP_LEN , batch_size, rnnt_config["in_feats"], dtype=torch.float16, device='cuda')
        feat_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * WARMUP_LEN 
        txt = torch.ones(batch_size, WARMUP_LEN , dtype=torch.int64, device='cuda')
        txt_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * WARMUP_LEN 
        dict_meta_data = train_preproc.get_packing_meta_data(feats.size(0), feat_lens, txt_lens)
        log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens, dict_meta_data)
        loss = loss_fn(log_probs, log_prob_lens, txt, txt_lens, dict_meta_data)
        loss /= args.grad_accumulation_steps
        del log_probs, log_prob_lens

        assert not torch.isnan(loss).any(), "should not have happened"
        if args.dist_lamb:
            optimizer._lazy_init_stage1()
            grad_scaler.scale(loss).backward()
            optimizer._lazy_init_stage2()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        optimizer.zero_grad()   # Don't really want to do the update
        if args.dist_lamb:
            optimizer.complete_reductions()
            optimizer.set_global_scale(grad_scaler._get_scale_async())
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

    if args.dist_lamb:
        optimizer_warm_up()

    logging.log_end(logging.constants.INIT_STOP)
    if multi_gpu:
        torch.distributed.barrier()
    logging.log_start(logging.constants.RUN_START)
    if multi_gpu:
        torch.distributed.barrier()

    if args.pre_sort_for_seq_split and not args.vectorized_sampler:
        raise NotImplementedError("Pre sort only works with vectorized sampler for now")
    logging.log_event(logging.constants.DATA_TRAIN_NUM_BUCKETS, value=args.num_buckets)

    if args.num_buckets is not None:
        if args.vectorized_sampler:
            builder = dali_sampler.VectorizedBucketingSampler
        else:
            builder = dali_sampler.BucketingSampler

        train_sampler = builder(
            train_dataset_kw,
            args.num_buckets,
            batch_size,
            world_size,
            args.epochs,
            sampler_seed,
            args.dist_sampler,
            args.pre_sort_for_seq_split
        )
    else:
        train_sampler = dali_sampler.SimpleSampler(train_dataset_kw)

    eval_sampler = dali_sampler.SimpleSampler(val_dataset_kw)

    train_sampler.sample(   file_names=args.train_manifests, 
                            in_mem_file_list=args.in_mem_file_list,
                            tokenized_transcript=args.tokenized_transcript)
    eval_sampler.sample(file_names=args.val_manifests, 
                        in_mem_file_list=args.in_mem_file_list,
                        tokenized_transcript=args.tokenized_transcript)


    # Setup DALI pipeline
    if args.synthetic_audio_seq_len is None and args.synthetic_text_seq_len is None:
        synthetic_seq_len = None
    elif args.synthetic_audio_seq_len is not None and args.synthetic_text_seq_len is not None:
        synthetic_seq_len = [args.synthetic_audio_seq_len, args.synthetic_text_seq_len]
    else:
        raise Exception("synthetic seq length for both text and audio need to be specified")
    train_loader = DaliDataLoader(gpu_id=args.local_rank,
                                  dataset_path=args.dataset_dir,
                                  config_data=train_dataset_kw,
                                  config_features=train_features_kw,
                                  json_names=args.train_manifests,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  grad_accumulation_steps=args.grad_accumulation_steps,
                                  pipeline_type="train",
                                  device_type=args.dali_device,
                                  tokenizer=tokenizer,
                                  num_threads=args.data_cpu_threads,
                                  synthetic_seq_len=synthetic_seq_len,
                                  seed=dali_seed,
                                  in_mem_file_list=args.in_mem_file_list,
                                  enable_prefetch=args.enable_prefetch,
                                  tokenized_transcript=args.tokenized_transcript,
                                  preproc=train_preproc,
                                  min_seq_split_len=args.min_seq_split_len,
                                  pre_sort=args.pre_sort_for_seq_split,
                                  jit_tensor_formation=args.jit_tensor_formation,
                                  dont_use_mmap=args.dali_dont_use_mmap)


    val_loader = DaliDataLoader(gpu_id=args.local_rank,
                                dataset_path=args.dataset_dir,
                                config_data=val_dataset_kw,
                                config_features=val_features_kw,
                                json_names=args.val_manifests,
                                batch_size=args.val_batch_size,
                                sampler=eval_sampler,
                                pipeline_type="val",
                                device_type=args.dali_device,
                                tokenizer=tokenizer,
                                num_threads=args.data_cpu_threads,
                                seed=dali_seed,
                                tokenized_transcript=args.tokenized_transcript,
                                in_mem_file_list=args.in_mem_file_list,
                                dont_use_mmap=args.dali_dont_use_mmap)


    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

    logging.log_event(logging.constants.TRAIN_SAMPLES, value=train_loader.dataset_size)
    logging.log_event(logging.constants.EVAL_SAMPLES, value=val_loader.dataset_size)

    logging.log_event(logging.constants.OPT_NAME, value='lamb')
    logging.log_event(logging.constants.OPT_BASE_LR, value=args.lr)
    logging.log_event(logging.constants.OPT_LAMB_EPSILON, value=opt_eps)
    logging.log_event(logging.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=args.lr_exp_gamma)
    logging.log_event(logging.constants.OPT_LR_WARMUP_EPOCHS, value=args.warmup_epochs)
    logging.log_event(logging.constants.OPT_LAMB_LR_HOLD_EPOCHS, value=args.hold_epochs)
    logging.log_event(logging.constants.OPT_LAMB_BETA_1, value=args.beta1)
    logging.log_event(logging.constants.OPT_LAMB_BETA_2, value=args.beta2)
    logging.log_event(logging.constants.OPT_GRADIENT_CLIP_NORM, value=args.clip_norm)
    logging.log_event(logging.constants.OPT_LR_ALT_DECAY_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LR_ALT_WARMUP_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LAMB_LR_MIN, value=args.min_lr)
    logging.log_event(logging.constants.OPT_WEIGHT_DECAY, value=args.weight_decay)


    # load checkpoint
    meta = {'best_wer': 10**6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'RNN-T',
                                args.keep_milestones, use_amp=True)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt

    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)


    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    last_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1

    # training loop
    model.train()
    if args.multi_tensor_ema == True:
        ema_model_weight_list, model_weight_list, overflow_buf_for_ema = init_multi_tensor_ema(optimizer, model, ema_model, args.ema_update_type)

    copy_stream = torch.cuda.Stream()
    pred_stream = torch.cuda.Stream()
    def buffer_pre_allocation():
        max_seq_len = math.ceil(train_preproc.audio_duration_to_seq_len(
                                                    cfg['input_train']['audio_dataset']['max_duration'], 
                                                    after_subsampling=False,
                                                    after_stack_time=False
                                                    ) 
                        * cfg["input_train"]["audio_dataset"]["speed_perturbation"]["max_rate"])
        max_txt_len = train_loader.data_iterator().max_txt_len
        print_once(f'Pre-allocate buffer with max_seq_len of %d and max_txt_len of %d' % (max_seq_len, max_txt_len))
        audio = torch.ones(batch_size, cfg["input_val"]["filterbank_features"]["n_filt"], max_seq_len, dtype=torch.float32, device='cuda')
        audio_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * max_seq_len
        txt = torch.ones(batch_size, max_txt_len, dtype=torch.int64, device='cuda')
        txt_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * max_txt_len
        feats, feat_lens = train_feat_proc([audio, audio_lens])
        if args.dist_lamb:
            feats = feats.half()

        meta_data = []
        B_split = batch_size // args.batch_split_factor
        for i in range(args.batch_split_factor):
            meta_data.append(train_preproc.get_packing_meta_data(   feats.size(0), 
                                                                    feat_lens[i*B_split:(i+1)*B_split], 
                                                                    txt_lens[i*B_split:(i+1)*B_split]))

        train_step( model, loss_fn, args, batch_size, feats, feat_lens, txt, txt_lens, optimizer, 
                    grad_scaler, meta_data, None, rnnt_graph, copy_stream, pred_stream)


    if args.buffer_pre_alloc:
        buffer_pre_allocation()

    training_start_time = time.time()
    training_utts = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):

        logging.log_start(logging.constants.BLOCK_START,
                          metadata=dict(first_epoch_num=epoch,
                                        epoch_count=1))
        logging.log_start(logging.constants.EPOCH_START,
                          metadata=dict(epoch_num=epoch))

        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()
        

        if args.enable_prefetch:
            train_loader.data_iterator().prefetch()

        step_start_time = time.time()

        for batch in train_loader:
            if accumulated_batches == 0:
                if not args.dist_lamb:
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
                adjust_lr(step, epoch)
                step_utts = 0
                all_feat_lens = []

            if args.enable_prefetch:
                # when prefetch is enabled, train_feat_proc at prefetch time
                feats, feat_lens, txt, txt_lens = batch
                meta_data = train_preproc.meta_data
            else:
                audio, audio_lens, txt, txt_lens = batch
                feats, feat_lens = train_feat_proc([audio, audio_lens])
                if args.dist_lamb:
                    feats = feats.half()
                meta_data = []
                B_split = batch_size // args.batch_split_factor
                for i in range(args.batch_split_factor):
                    meta_data.append(train_preproc.get_packing_meta_data(   feats.size(0), 
                                                                            feat_lens[i*B_split:(i+1)*B_split], 
                                                                            txt_lens[i*B_split:(i+1)*B_split]))
                        
            if args.enable_seq_len_stats:
                all_feat_lens += feat_lens



            loss_item, lr_item = train_step( model, loss_fn, args, batch_size, feats, feat_lens, txt, txt_lens, optimizer, 
                                    grad_scaler, meta_data, train_loader, rnnt_graph, copy_stream, pred_stream)


            step_utts += txt_lens.size(0) * world_size
            epoch_utts += txt_lens.size(0) * world_size
            accumulated_batches += 1
            if accumulated_batches % args.grad_accumulation_steps == 0:

                if args.dist_lamb:
                    optimizer.complete_reductions()
                    optimizer.set_global_scale(grad_scaler._get_scale_async())
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                else:
                    optimizer.step()

                if args.multi_tensor_ema == True:
                    apply_multi_tensor_ema(model_weight_list, ema_model_weight_list, args.ema, overflow_buf_for_ema)
                else:
                    apply_ema(optimizer, ema_model, args.ema)

                if step % args.log_frequency == 0:

                    if args.prediction_frequency is None or step % args.prediction_frequency == 0:
                        preds = greedy_decoder.decode(feats.half(), feat_lens)
                        wer, pred_utt, ref = greedy_wer(
                                preds,
                                txt,
                                txt_lens,
                                tokenizer.detokenize)
                        print_once(f'  Decoded:   {pred_utt[:90]}')
                        print_once(f'  Reference: {ref[:90]}')
                        wer = {'wer': 100 * wer}
                    else:
                        wer = {}

                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    dict_log = {'loss': loss_item,
                                 **wer,  # optional entry
                                'throughput': step_utts / step_time,
                                'took': step_time,
                                'lrate': optimizer._lr.item() if args.dist_lamb else optimizer.param_groups[0]['lr']} # TODO: eliminate sync

                    if args.enable_seq_len_stats:
                        dict_log["seq-len-min"] = min(all_feat_lens).item()
                        dict_log["seq-len-max"] = max(all_feat_lens).item()

                    log((epoch, step % steps_per_epoch or steps_per_epoch, steps_per_epoch),
                        step, 'train', dict_log)



                step += 1
                accumulated_batches = 0
                # end of step

        logging.log_end(logging.constants.EPOCH_STOP,
                        metadata=dict(epoch_num=epoch))

        epoch_time = time.time() - epoch_start_time
        log((epoch,), None, 'train_avg', {'throughput': epoch_utts / epoch_time,
                                          'took': epoch_time})
        # logging throughput for dashboard
        logging.log_event(key='throughput', value= epoch_utts / epoch_time)

        if epoch % args.val_frequency == 0:
            wer = evaluate(epoch, step, val_loader, val_feat_proc,
                           tokenizer.detokenize, ema_model, loss_fn,
                           greedy_decoder, args.amp_level)

            last_wer = wer
            if wer < best_wer and epoch >= args.save_best_from:
                checkpointer.save(model, ema_model, optimizer, epoch,
                                  step, best_wer, is_best=True)
                best_wer = wer

        save_this_epoch = (args.save_frequency is not None and epoch % args.save_frequency == 0) \
                       or (epoch in args.keep_milestones)
        if save_this_epoch:
            checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)

        training_utts += epoch_utts
        logging.log_end(logging.constants.BLOCK_STOP, metadata=dict(first_epoch_num=epoch))

        if last_wer <= args.target:
            logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'success'})
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        # end of epoch


    training_time = time.time() - training_start_time
    log((), None, 'train_avg', {'throughput': training_utts / training_time})

    if last_wer > args.target:
        logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'aborted'})

    if epoch == args.epochs:
        evaluate(epoch, step, val_loader, val_feat_proc, tokenizer.detokenize,
                 ema_model, loss_fn, greedy_decoder, args.amp_level)

    flush_log()
    if args.save_at_the_end:
        checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)


if __name__ == "__main__":
    main()

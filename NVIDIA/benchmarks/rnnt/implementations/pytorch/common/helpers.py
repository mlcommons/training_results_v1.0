# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

import glob
import os
import re
from collections import OrderedDict

from apex import amp

import torch
import torch.distributed as dist

from .metrics import word_error_rate
import math


def __rnnt_decoder_predictions_tensor(tensor, detokenize):
    """
    Takes output of greedy rnnt decoder and converts to strings.
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    return [detokenize(pred) for pred in tensor]


def print_once(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)


def greedy_wer(preds, tgt, tgt_lens, detokenize):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints wer and prediction examples to screen
    Args:
        tensors: A list of 3 tensors (predictions, targets, target_lengths)
        labels: A list of labels

    Returns:
        word error rate
    """
    with torch.no_grad():
        references = gather_transcripts([tgt], [tgt_lens], detokenize)
        hypotheses = __rnnt_decoder_predictions_tensor(preds, detokenize)

    wer, _, _ = word_error_rate(hypotheses, references)
    return wer, hypotheses[0], references[0]


def gather_losses(losses_list):
    return [torch.mean(torch.stack(losses_list))]


def gather_predictions(predictions_list, detokenize):
    rnnt_predictions = (
        __rnnt_decoder_predictions_tensor(prediction, detokenize)
        for prediction in predictions_list
    )

    return [
        prediction
        for batch in rnnt_predictions
        for prediction in batch
    ]


def gather_transcripts(transcript_list, transcript_len_list, detokenize):
    return [
        detokenize(t[:l].long().cpu().numpy().tolist())
        for txt, lens in zip(transcript_list, transcript_len_list)
        for t, l in zip(txt, lens)
    ]


def process_evaluation_epoch(aggregates):
    """
    Processes results from each worker at the end of evaluation and combine to final result
    Args:
        aggregates: dictionary containing information of entire evaluation
    Return:
        wer: final word error rate
        loss: final loss
    """
    if 'losses' in aggregates:
        eloss = torch.mean(torch.stack(aggregates['losses'])).item()
    else:
        eloss = None

    hypotheses = aggregates['preds']
    references = aggregates['txts']

    wer, scores, num_words = word_error_rate(hypotheses, references)
    multi_gpu = dist.is_initialized()
    if multi_gpu:
        if eloss is not None:
            eloss /= dist.get_world_size()
            eloss_tensor = torch.tensor(eloss).cuda()
            dist.all_reduce(eloss_tensor)
            eloss = eloss_tensor.item()

        scores_tensor = torch.tensor(scores).cuda()
        dist.all_reduce(scores_tensor)
        scores = scores_tensor.item()
        num_words_tensor = torch.tensor(num_words).cuda()
        dist.all_reduce(num_words_tensor)
        num_words = num_words_tensor.item()
        wer = scores * 1.0 / num_words
    return wer, eloss


def num_weights(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class Checkpointer(object):

    def __init__(self, save_dir, model_name, keep_milestones=[100,200,300],
                 use_amp=False):
        self.save_dir = save_dir
        self.keep_milestones = keep_milestones
        self.use_amp = use_amp
        self.model_name = model_name

        tracked = [
            (int(re.search('epoch(\d+)_', f).group(1)), f)
            for f in glob.glob(f'{save_dir}/{self.model_name}_epoch*_checkpoint.pt')]
        tracked = sorted(tracked, key=lambda t: t[0])
        self.tracked = OrderedDict(tracked)

    def save(self, model, ema_model, optimizer, epoch, step, best_wer,
             is_best=False):
        """Saves model checkpoint for inference/resuming training.

        Args:
            model: the model, optionally wrapped by DistributedDataParallel
            ema_model: model with averaged weights, can be None
            optimizer: optimizer
            epoch (int): epoch during which the model is saved
            step (int): number of steps since beginning of training
            best_wer (float): lowest recorded WER on the dev set
            is_best (bool, optional): set name of checkpoint to 'best'
                and overwrite the previous one
        """
        rank = 0
        if dist.is_initialized():
            dist.barrier()
            rank = dist.get_rank()

        if rank != 0:
            return

        # Checkpoint already saved
        if not is_best and epoch in self.tracked:
            return

        unwrap_ddp = lambda model: getattr(model, 'module', model)
        state = {
            'epoch': epoch,
            'step': step,
            'best_wer': best_wer,
            'state_dict': unwrap_ddp(model).state_dict(),
            'ema_state_dict': unwrap_ddp(ema_model).state_dict() if ema_model is not None else None,
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict() if self.use_amp else None,
        }

        if is_best:
            fpath = os.path.join(
                self.save_dir, f"{self.model_name}_best_checkpoint.pt")
        else:
            fpath = os.path.join(
                self.save_dir, f"{self.model_name}_epoch{epoch}_checkpoint.pt")

        print_once(f"Saving {fpath}...")
        torch.save(state, fpath)

        if not is_best:
            # Remove old checkpoints; keep milestones and the last two
            self.tracked[epoch] = fpath
            for epoch in set(list(self.tracked)[:-2]) - set(self.keep_milestones):
                try:
                    os.remove(self.tracked[epoch])
                except:
                    pass
                del self.tracked[epoch]

    def last_checkpoint(self):
        tracked = list(self.tracked.values())

        if len(tracked) >= 1:
            try:
                torch.load(tracked[-1], map_location='cpu')
                return tracked[-1]
            except:
                print_once(f'Last checkpoint {tracked[-1]} appears corrupted.')

        elif len(tracked) >= 2:
            return tracked[-2]
        else:
            return None

    def load(self, fpath, model, ema_model, optimizer, meta):

        print_once(f'Loading model from {fpath}')
        checkpoint = torch.load(fpath, map_location="cpu")

        unwrap_ddp = lambda model: getattr(model, 'module', model)
        state_dict = checkpoint['state_dict']
        unwrap_ddp(model).load_state_dict(state_dict, strict=False)

        if ema_model is not None:
            if checkpoint.get('ema_state_dict') is not None:
                key = 'ema_state_dict'
            else:
                key = 'state_dict'
                print_once('WARNING: EMA weights not found in the checkpoint.')
                print_once('WARNING: Initializing EMA model with regular params.')
            state_dict = checkpoint[key]
            unwrap_ddp(ema_model).load_state_dict(state_dict, strict=False)

        optimizer.load_state_dict(checkpoint['optimizer'])

        if self.use_amp:
            amp.load_state_dict(checkpoint['amp'])

        meta['start_epoch'] = checkpoint.get('epoch')
        meta['best_wer'] = checkpoint.get('best_wer', meta['best_wer'])

class Preproc():
    def __init__(self, feat_proc, dist_lamb, apex_transducer_joint, batch_split_factor, cfg):
        self.feat_proc = feat_proc
        self.dist_lamb = dist_lamb
        self.apex_transducer_joint = apex_transducer_joint
        self.enc_stack_time_factor = cfg["rnnt"]["enc_stack_time_factor"]
        self.window_stride = cfg["input_val"]["filterbank_features"]["window_stride"]
        self.frame_subsampling =  cfg["input_val"]["frame_splicing"]["frame_subsampling"]
        self.batch_split_factor = batch_split_factor
        self.list_packed_batch_cpu = [torch.tensor(0, dtype=torch.int64, device='cpu').pin_memory() 
                                        for i in range(self.batch_split_factor)]

    def preproc_func(self, audio, audio_shape):
        audio_, audio_shape_ = self.feat_proc([audio, audio_shape])
        if self.dist_lamb:
            audio_ = audio_.half()

        return audio_, audio_shape_

    def get_meta_data(self, max_f_len, audio_shape_, transcripts, transcripts_lengths, async_cp=False, idx=0):
        self.meta_data = []
        B_split = transcripts.size(0) // self.batch_split_factor
        for i in range(self.batch_split_factor):
            self.meta_data.append(self.get_packing_meta_data(   max_f_len, 
                                                                audio_shape_[i*B_split:(i+1)*B_split], 
                                                                transcripts_lengths[i*B_split:(i+1)*B_split],
                                                                async_cp,
                                                                i))

    def get_packing_meta_data(self, max_f_len, feat_lens, txt_lens, async_cp=False, idx=0):
        dict_meta_data = {"batch_offset": None,
                          "g_len": None,
                          "max_f_len": None,
                          "packed_batch": None}
        if self.apex_transducer_joint is not None:
            g_len = txt_lens + 1
            if self.apex_transducer_joint == "pack":
                batch_offset = torch.cumsum(g_len * ((feat_lens+self.enc_stack_time_factor-1)//self.enc_stack_time_factor), dim=0)
                if async_cp:
                    # copy packed batch size asyncly for later use
                    self.list_packed_batch_cpu[idx].copy_(batch_offset[-1].detach(), non_blocking=True)

                dict_meta_data = {"batch_offset": batch_offset,
                                  "g_len": g_len,
                                  "max_f_len": (max_f_len+self.enc_stack_time_factor-1)//self.enc_stack_time_factor,
                                  "packed_batch": batch_offset[-1].item() if not async_cp else None}


            elif self.apex_transducer_joint == "not_pack":
                dict_meta_data["g_len"] = g_len
                dict_meta_data["packed_batch"] = 0
        return dict_meta_data

    def audio_duration_to_seq_len(self, audio_duration, after_subsampling, after_stack_time):
        if after_stack_time:
            assert after_subsampling, "after_stacktime == True while after_subsampling == False is not a valid use case"
        seq_len = audio_duration // self.window_stride + 1
        if after_subsampling == False:
            return seq_len
        else:
            seq_len_sub_sampled = math.ceil(seq_len / self.frame_subsampling)
            if after_stack_time == False:
                return seq_len_sub_sampled
            else:
                return (seq_len_sub_sampled + self.enc_stack_time_factor - 1) // self.enc_stack_time_factor

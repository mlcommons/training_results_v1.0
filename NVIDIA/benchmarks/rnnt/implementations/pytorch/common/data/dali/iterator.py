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

import torch
import torch.distributed as dist
import numpy as np
from common.helpers import print_once
from common.text import _clean_text, punctuation_map


def normalize_string(s, charset, punct_map):
    """Normalizes string.

    Example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'
    """
    charset = set(charset)
    try:
        text = _clean_text(s, ["english_cleaners"], punct_map).strip()
        return ''.join([tok for tok in text if all(t in charset for t in tok)])
    except:
        print(f"WARNING: Normalizing failed: {s}")
        return None


class DaliRnntIterator(object):
    """
    Returns batches of data for RNN-T training:
    preprocessed_signal, preprocessed_signal_length, transcript, transcript_length

    This iterator is not meant to be the entry point to Dali processing pipeline.
    Use DataLoader instead.
    """

    def __init__(   self, dali_pipelines, transcripts, tokenizer, batch_size, shard_size, pipeline_type, 
                    normalize_transcripts=False, synthetic_text_seq_len=None, enable_prefetch=False,
                    tokenized_transcript=False,
                    preproc=None, min_seq_split_len=-1, jit_tensor_formation=False):
        self.normalize_transcripts = normalize_transcripts
        self.tokenizer = tokenizer
        self.tokenized_transcript = tokenized_transcript
        self.batch_size = batch_size
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy

        # in train pipeline shard_size is set to divisable by batch_size, so PARTIAL policy is safe
        if pipeline_type == 'val':
            self.dali_it = DALIGenericIterator(
                dali_pipelines, ["audio", "label", "audio_shape"], reader_name="Reader",
                dynamic_shape=True, auto_reset=True,
                last_batch_policy=LastBatchPolicy.PARTIAL)
        else:
            self.dali_it = DALIGenericIterator(
                dali_pipelines, ["audio", "label", "audio_shape"], size=shard_size,
                dynamic_shape=True, auto_reset=True,
            )

        self.jit_tensor_formation = jit_tensor_formation
        self.tokenize(transcripts)
        self.synthetic_text_seq_len = synthetic_text_seq_len
        self.enable_prefetch = enable_prefetch
        self.prefetch_stream = torch.cuda.Stream()
        self.preproc = preproc
        self.pipeline_type = pipeline_type
        self.min_seq_split_len = min_seq_split_len
        self.pivot_len_cpu = torch.tensor(0, dtype=torch.int, device='cpu').pin_memory()      


    def tokenize(self, transcripts):
        transcripts = [transcripts[i] for i in range(len(transcripts))]
        if self.normalize_transcripts:
            transcripts = [
                normalize_string(
                    t,
                    self.tokenizer.charset,
                    punctuation_map(self.tokenizer.charset)
                ) for t in transcripts
            ]
        if not self.tokenized_transcript:
            transcripts = [self.tokenizer.tokenize(t) for t in transcripts]

        if self.jit_tensor_formation:
            self.tr = transcripts
        else:
            self.tr = np.empty(len(transcripts), dtype=object)
            for i in range(len(transcripts)):
                self.tr[i] = torch.tensor(transcripts[i])

        self.t_sizes = torch.tensor([len(t) for t in transcripts], dtype=torch.int32)
        self.max_txt_len = self.t_sizes.max().item()

    def _gen_transcripts(self, labels, normalize_transcripts: bool = True):
        """
        Generate transcripts in format expected by NN
        """
        ids = labels.flatten().numpy()
        transcripts = [torch.tensor(self.tr[i]) for i in ids] if self.jit_tensor_formation else self.tr[ids]
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True)

        return transcripts.cuda(), self.t_sizes[ids].cuda()

    def fetch_next(self):
        data = self.dali_it.__next__()
        audio, audio_shape = data[0]["audio"], data[0]["audio_shape"][:, 1]
        if audio.shape[0] == 0:
            # empty tensor means, other GPUs got last samples from dataset
            # and this GPU has nothing to do; calling `__next__` raises StopIteration
            return self.dali_it.__next__()
        if self.pipeline_type == 'val':
            audio = audio[:, :, :audio_shape.max()] # the last batch
        transcripts, transcripts_lengths = self._gen_transcripts(data[0]["label"])

        if self.synthetic_text_seq_len != None:
            transcripts = torch.randint(transcripts.max(), (transcripts.size(0), self.synthetic_text_seq_len), device=transcripts.device, dtype=transcripts.dtype)
            transcripts_lengths = torch.ones_like(transcripts_lengths) * self.synthetic_text_seq_len

        if self.enable_prefetch and self.preproc is not None:
            # prefeth path, do preprocessing here
            audio, audio_shape = self.preproc.preproc_func(audio, audio_shape)
            max_f_len = audio.size(0)
            if self.pipeline_type == "train" and self.min_seq_split_len > 0 and self.enable_prefetch:
                # seq split path, need to sort seq
                audio, audio_shape, transcripts, transcripts_lengths = self._prepare_seq_split(audio, audio_shape, transcripts, transcripts_lengths)

            # use async copy as we don't really want to sync here
            self.preproc.get_meta_data(max_f_len, audio_shape, transcripts, transcripts_lengths, async_cp=True)

        return audio, audio_shape, transcripts, transcripts_lengths


    def _prepare_seq_split(self, audio, audio_shape, transcripts, transcripts_lengths):
        idx_sorted = torch.argsort(audio_shape, descending=True)
        audio_shape_sorted = audio_shape[idx_sorted]
        audio_sorted = audio[:, idx_sorted]
        transcripts_sorted = transcripts[idx_sorted]
        transcripts_lengths_sorted = transcripts_lengths[idx_sorted]
        batch_size = audio_shape.size(0)
        self.split_batch_size = batch_size // 2  # currently only split once
        stack_factor = self.preproc.enc_stack_time_factor
        # make sure the first segment is multiple of stack_factor so that stack time can be done easily
        pivot_len = (audio_shape_sorted[self.split_batch_size] + stack_factor-1) // stack_factor * stack_factor
        # copy pivot len asyncly for later use
        self.pivot_len_cpu.copy_(pivot_len.detach(), non_blocking=True)
        return audio_sorted, audio_shape_sorted, transcripts_sorted, transcripts_lengths_sorted

    def __next__(self):
        if self.enable_prefetch:
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)
            # make sure all async copies are committed
            self.prefetch_stream.synchronize()
            if self.prefetched_data is None:
                raise StopIteration
            else:
                # write asyncly copied packed batch info into meta data
                for i, packed_batch_cpu in enumerate(self.preproc.list_packed_batch_cpu):
                    self.preproc.meta_data[i]["packed_batch"] = packed_batch_cpu.item()

                if self.pipeline_type == "train" and self.min_seq_split_len > 0:
                    # seq split path
                    audio, audio_shape, transcripts, transcripts_lengths = self.prefetched_data 
                    second_segment_len = audio.size(0) - self.pivot_len_cpu
                    if second_segment_len >= self.min_seq_split_len:
                        list_audio = [audio[:self.pivot_len_cpu], audio[self.pivot_len_cpu:, :self.split_batch_size]]
                        return list_audio, audio_shape, transcripts, transcripts_lengths
                
                # normal path
                return self.prefetched_data
        else:
            return self.fetch_next()

    def prefetch(self):
        with torch.cuda.stream(self.prefetch_stream):
            try:
                self.prefetched_data = self.fetch_next()
            except StopIteration:
                self.prefetched_data = None

    def __iter__(self):
        return self



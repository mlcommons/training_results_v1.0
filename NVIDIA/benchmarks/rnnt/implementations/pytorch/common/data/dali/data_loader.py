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

import os
import math
import multiprocessing
import numpy as np
import torch.distributed as dist
from .iterator import DaliRnntIterator
from .pipeline import DaliPipeline
from common.helpers import print_once


class DaliDataLoader:
    """
    DataLoader is the main entry point to the data preprocessing pipeline.
    To use, create an object and then just iterate over `data_iterator`.
    DataLoader will do the rest for you.
    Example:
        data_layer = DataLoader(DaliTrainPipeline, path, json, bs, ngpu)
        data_it = data_layer.data_iterator
        for data in data_it:
            print(data)  # Here's your preprocessed data

    Args:
        device_type: Which device to use for preprocessing. Choose: "cpu", "gpu"
        pipeline_type: Choose: "train", "val"
    """

    def __init__(self, gpu_id, dataset_path: str, config_data: dict, config_features: dict, json_names: list,
                 tokenizer, batch_size: int, sampler, pipeline_type: str, seed, grad_accumulation_steps: int = 1,
                 num_threads=multiprocessing.cpu_count(),
                 tokenized_transcript=False,
                 device_type: str = "gpu", synthetic_seq_len=None, 
                 in_mem_file_list=True, enable_prefetch=False, preproc=None, min_seq_split_len=-1,
                 pre_sort=False, jit_tensor_formation=False, dont_use_mmap=False):

        import torch
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.drop_last = (pipeline_type == 'train')
        self.device_type = device_type
        self.pipeline_type = self._parse_pipeline_type(pipeline_type)
        self.sampler = sampler
        self._dali_data_iterator = self._init_iterator(gpu_id=gpu_id, dataset_path=dataset_path,
                                                       config_data=config_data,
                                                       config_features=config_features,
                                                       json_names=json_names, tokenizer=tokenizer,
                                                       num_threads=num_threads,
                                                       pipeline_type=pipeline_type,
                                                       synthetic_seq_len=synthetic_seq_len, seed=seed,
                                                       in_mem_file_list=in_mem_file_list,
                                                       tokenized_transcript=tokenized_transcript,
                                                       enable_prefetch=enable_prefetch, preproc=preproc,
                                                       min_seq_split_len=min_seq_split_len,
                                                       pre_sort=pre_sort,
                                                       jit_tensor_formation=jit_tensor_formation,
                                                       dont_use_mmap=dont_use_mmap)

    def _init_iterator(self, gpu_id, dataset_path, config_data, config_features, json_names: list, tokenizer: list,
                       num_threads, pipeline_type, synthetic_seq_len, seed, in_mem_file_list, enable_prefetch, preproc,
                       tokenized_transcript=False, min_seq_split_len=-1, pre_sort=False, jit_tensor_formation=False,
                       dont_use_mmap=False):

        """
        Returns data iterator. Data underneath this operator is preprocessed within Dali
        """
        if in_mem_file_list:
            assert (len(self.sampler.files) > 0 and len(self.sampler.labels) > 0), "Please run sampler.sample() first"
        else:
            assert self.sampler.file_list_path is not None, "Please run sampler.sample() first"
        self.dataset_size = self.sampler.get_dataset_size()
        print_once(f"Dataset read by DALI. Number of samples: {self.dataset_size}")

        pipeline = DaliPipeline.from_config(config_data=config_data, config_features=config_features, device_id=gpu_id,
                                            file_root=dataset_path, sampler=self.sampler,
                                            device_type=self.device_type, batch_size=self.batch_size,
                                            pipeline_type=pipeline_type,
                                            num_cpu_threads=num_threads,
                                            synthetic_seq_len=synthetic_seq_len, seed=seed,
                                            in_mem_file_list=in_mem_file_list,
                                            pre_sort=pre_sort, dont_use_mmap=dont_use_mmap)

        return DaliRnntIterator([pipeline], transcripts=self.sampler.transcripts, tokenizer=tokenizer, batch_size=self.batch_size,
                                  shard_size=self._shard_size(), pipeline_type=pipeline_type,
                                  synthetic_text_seq_len=synthetic_seq_len[1] if synthetic_seq_len is not None else None,
                                  tokenized_transcript=tokenized_transcript,
                                  enable_prefetch=enable_prefetch, preproc=preproc, min_seq_split_len=min_seq_split_len,
                                  jit_tensor_formation=jit_tensor_formation)

    @staticmethod
    def _parse_pipeline_type(pipeline_type):
        pipe = pipeline_type.lower()
        assert pipe in ("train", "val"), 'Invalid pipeline type ("train", "val").'
        return pipe

    def _shard_size(self):
        """
        Total number of samples handled by a single GPU in a single epoch.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.drop_last:
            divisor = world_size * self.batch_size * self.grad_accumulation_steps
            return self.dataset_size // divisor * divisor // world_size
        else:
            return int(math.ceil(self.dataset_size / world_size))

    def __len__(self):
        """
        Number of batches handled by each GPU.
        """
        if self.drop_last:
            assert self._shard_size() % self.batch_size == 0, f'{self._shard_size()} {self.batch_size}'

        return int(math.ceil(self._shard_size() / self.batch_size))

    def data_iterator(self):
        return self._dali_data_iterator

    def __iter__(self):
        return self._dali_data_iterator

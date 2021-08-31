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

import numpy as np
import torch

from common.helpers import print_once


def hash_list_of_strings(li):
    return str(abs(hash(''.join(li))))

def _parse_json(json_path: str, start_label=0, predicate=lambda json: True, tokenized_transcript=False):
    """
    Parses json file to the format required by DALI
    Args:
        json_path: path to json file
        start_label: the label, starting from which DALI will assign consecutive int numbers to every transcript
        predicate: function, that accepts a sample descriptor (i.e. json dictionary) as an argument.
                   If the predicate for a given sample returns True, it will be included in the dataset.

    Returns:
        output_files: dictionary, that maps file name to label assigned by DALI
        transcripts: dictionary, that maps label assigned by DALI to the transcript
    """
    import json
    global cnt
    with open(json_path) as f:
        librispeech_json = json.load(f)
    output_files = {}
    transcripts = {}
    curr_label = start_label
    for original_sample in librispeech_json:
        if not predicate(original_sample):
            continue
        transcripts[curr_label] = original_sample['tokenized_transcript' if tokenized_transcript else 'transcript']
        output_files[original_sample['files'][-1]['fname']] = dict(
            label=curr_label,
            duration=original_sample['original_duration'],
        )
        curr_label += 1
    return output_files, transcripts

def _parse_pkl(pkl_path: str, start_label=0, predicate=lambda pkl: True, tokenized_transcript=True):
    if not tokenized_transcript:
        raise NotImplementedError("pickle input only works with tokenized_transcript")
    import pickle
    with open(pkl_path, 'rb') as f:
        librispeech_pkl = pickle.load(f)
    output_files = {}
    transcripts = {}
    curr_label = start_label
    for original_sample in librispeech_pkl:
        if not predicate(original_sample):
            continue
        transcripts[curr_label] = original_sample['tokenized_transcript']
        output_files[original_sample['fname']] = dict(
            label=curr_label,
            duration=original_sample['original_duration'],
        )
        curr_label += 1
    return output_files, transcripts

class SimpleSampler:
    def __init__(self, config_data, dist_sampler=False):
        self.file_list_path = None
        self.files, self.labels = [], []
        self.dataset_size = None
        self.dist_sampler = dist_sampler
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.config_data = config_data

    def write_file_list(self, names, labels):
        with open(self.file_list_path, 'w') as f:
            f.writelines(f'{name} {label}\n' for name, label in zip(names, labels))

    def get_file_list_path(self):
        assert self.file_list_path, 'File list not initialized. Run make_file_list first'
        return self.file_list_path

    def get_dataset_size(self):
        assert self.dataset_size, 'Dataset size not known. Run make_file_list first'
        return self.dataset_size

    def is_sampler_random(self):
        return False

    def process_output_files(self, output_files):
        print_once('Launching simple sampler')
        self.dataset_size = len(output_files)
        self.max_duration = max(entry['duration'] for _, entry in output_files.items())
        return [path           for path, _     in output_files.items()], \
               [entry['label'] for _,    entry in output_files.items()]

    def make_file_list(self, output_files, json_names):
        file_name = hash_list_of_strings(json_names)
        if self.dist_sampler:
            file_name += '__%d' % self.rank
        self.file_list_path = os.path.join(
            "/tmp",
            "rnnt_dali.file_list." + file_name
        )
        self.write_file_list(*self.process_output_files(output_files))

    def make_files(self, output_files):
        self.files, self.labels = self.process_output_files(output_files)

    def sample(self, file_names, in_mem_file_list, tokenized_transcript):
        output_files, self.transcripts = {}, {}
        max_duration = self.config_data['max_duration']
        for file in file_names:
            if file.endswith('.json'):
                parse_func = _parse_json
            elif file.endswith('.pkl'):
                parse_func = _parse_pkl
            else:
                raise NotImplementedError("Please supply supported input data file type: json or pickle")
            of, tr = parse_func(
                file if file[0] == '/' else os.path.join(dataset_path, file),
                len(output_files),
                predicate=lambda file: file['original_duration'] <= max_duration,
                tokenized_transcript=tokenized_transcript,
            )
            output_files.update(of)
            self.transcripts.update(tr)

        if in_mem_file_list:
            self.make_files(output_files)
        else:
            self.make_file_list(output_files, file_names)


class BucketingSampler(SimpleSampler):
    def __init__(self, config_data, num_buckets, batch_size, num_workers, num_epochs, seed, dist_sampler, pre_sort):
        super(BucketingSampler, self).__init__(config_data, dist_sampler)
        assert not pre_sort, "pre_sort not supported in BucketingSampler"
        self.rng = np.random.default_rng(seed=seed)
        self.num_buckets = num_buckets
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def process_output_files(self, output_files):
        print_once('Launching bucketing sampler')
        names = list(output_files)
        lengths = [output_files[name]['duration'] for name in names]
        labels = np.array([output_files[name]['label'] for name in names])
        len_ids = np.argsort(lengths)
        buckets = np.array_split(len_ids, self.num_buckets)

        gbs = self.batch_size * self.num_workers
        shuffled_buckets = np.array([
            perm
            for _ in range(self.num_epochs)          # for every epoch
            for bucket in buckets                    # from every bucket
            for perm in self.rng.permutation(bucket) # pick samples in random order
        ])

        # drop last batch
        epochs = np.reshape(shuffled_buckets, [self.num_epochs, -1])
        to_drop = epochs.shape[1] - (epochs.shape[1] // gbs * gbs)
        for epoch in epochs:
            dropped_idxs = self.rng.choice(epochs.shape[1], to_drop, replace=False)
            if to_drop > 0:
                epoch[dropped_idxs] = -1
        epochs = epochs[epochs != -1].reshape(self.num_epochs, -1)
        self.dataset_size = epochs.shape[1]

        epochs_iters_batch = np.reshape(epochs, [self.num_epochs, -1, gbs])

        # shuffle iterations in epochs perserving batches
        for epoch in epochs_iters_batch:
            self.rng.shuffle(epoch, axis=0)

        epochs_iters_batch_worker = np.reshape(
            epochs_iters_batch,
            [self.num_epochs, -1, self.batch_size, self.num_workers]
        )
        workers_epochs_iters_batch = np.moveaxis(epochs_iters_batch_worker, -1, 0)

        if self.dist_sampler:
            order = workers_epochs_iters_batch[self.rank].flatten()
        else:
            order = workers_epochs_iters_batch.flatten()

        return np.array(names) [order].tolist(), \
               np.array(labels)[order].tolist()


    def is_sampler_random(self):
        return True


class VectorizedBucketingSampler(SimpleSampler):
    def __init__(self, config_data, num_buckets, batch_size, num_workers, num_epochs, seed, dist_sampler, pre_sort):
        super(VectorizedBucketingSampler, self).__init__(config_data, dist_sampler)
        self.seed = seed
        self.num_buckets = num_buckets
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_sort = pre_sort

    def process_output_files(self, output_files):
        print_once('Launching vectorized bucketing sampler')
        names = list(output_files)
        lengths = [output_files[name]['duration'] for name in names]
        labels = np.array([output_files[name]['label'] for name in names])

        dur = torch.tensor(lengths, device='cuda')
        len_ids = dur.argsort()
        buckets = len_ids.tensor_split(self.num_buckets)
        padded_buckets = torch.nn.utils.rnn.pad_sequence(buckets, padding_value=-1, batch_first=True)

        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(self.seed)
            self.seed += 1

            buckets_shuffler = torch.rand(self.num_epochs, *padded_buckets.shape, device='cuda')
            shuffle_columnvise = buckets_shuffler.argsort(dim=2)
            epochs, num_buckets, samples = shuffle_columnvise.shape
            shift = torch.arange(0, samples*num_buckets, samples, device='cuda').view(1, -1, 1)
            shuffle_globalvise = shuffle_columnvise + shift

            shuffled_buckets = padded_buckets.take(shuffle_globalvise)

            gbs = self.batch_size * self.num_workers
            unpadded = shuffled_buckets[shuffled_buckets != -1].view(epochs, -1)
            epochs, samples = unpadded.shape

            to_drop = samples - (samples // gbs * gbs)
            mask = torch.ones_like(unpadded, dtype=bool, device='cuda')
            removed_samples = torch.rand(unpadded.shape, device='cuda').argsort(dim=1)[:, :to_drop]
            epoch_idx = torch.arange(self.num_epochs).view(-1, 1).expand(self.num_epochs, to_drop)
            mask[epoch_idx.flatten(), removed_samples.flatten()] = False

            batch_aligned = unpadded[mask].view(self.num_epochs, -1, self.batch_size)
            _, num_iterations, _ = batch_aligned.shape

            epochs, num_batches, bs = batch_aligned.view(self.num_epochs, -1, gbs).shape
            new_order = torch.rand(epochs, num_batches, device='cuda')
            nwo = new_order.argsort(dim=1).view(-1, num_batches, 1) * bs \
                + torch.arange(0, bs, 1, device='cuda').view(1,1,-1) \
                + torch.arange(0, epochs*num_batches*bs, num_batches*bs,device='cuda').view(-1, 1, 1)

            out = batch_aligned.take(nwo)
            if self.pre_sort:
                # At this point, the mini-batch has been formed. Now we can arrange work to each GPU
                pert_range = self.config_data['speed_perturbation']['max_rate'] - self.config_data['speed_perturbation']['min_rate']
                self.pert_coeff = torch.rand(out.size(0), out.size(1), out.size(2), device="cuda") * pert_range + self.config_data['speed_perturbation']['min_rate']
                dur_after_pert = dur[out] * self.pert_coeff
                idx_asc = dur_after_pert.argsort(dim=-1)
                idx_des = torch.flip(idx_asc, dims=[-1])
                idx_mix = torch.ones_like(idx_asc)
                # Assuming batch size is a multiple of 2.
                idx_mix[:, :, ::2] = idx_asc[:, :, :idx_asc.size(-1) // 2]
                idx_mix[:, :, 1::2] = idx_des[:, :, :idx_des.size(-1) // 2]

                out = torch.gather(out, 2, idx_mix)
                self.pert_coeff = torch.gather(self.pert_coeff, 2, idx_mix)

                # to test, try 
                # dur[out] * self.pert_coeff
        

        if self.dist_sampler:
            out = out.view(epochs, -1, self.num_workers, self.batch_size).moveaxis(2, 0)
            out = out[self.rank]
            if self.pre_sort:
                self.pert_coeff = self.pert_coeff.view(epochs, -1, self.num_workers, self.batch_size).moveaxis(2, 0)
                self.pert_coeff = self.pert_coeff[self.rank].cpu() 


        self.dataset_size = num_iterations * self.batch_size
        out = out.cpu()
        
        return np.array(names) [out.flatten()].tolist(), \
               np.array(labels)[out.flatten()].tolist()

    def is_sampler_random(self):
        return True


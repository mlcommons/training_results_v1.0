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

"""
create train or eval dataset.
"""
import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2

from PIL import Image
from io import BytesIO

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def create_dataset(dataset_path,
                   do_train, 
                   image_size=224,
                   crop_min=0.08,
                   repeat_num=1,
                   batch_size=32,
                   num_workers=12):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    device_num = int(os.getenv("RANK_SIZE"))
    rank_id = int(os.getenv('RANK_ID'))

    if do_train:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers, shuffle=True,
                                    num_shards=device_num, shard_id=rank_id)
    else:
        batch_per_step = batch_size * device_num
        print("eval batch per step:{}".format(batch_per_step))
        if batch_per_step < 50000:
            if 50000 % batch_per_step == 0:
                num_padded = 0
            else:
                num_padded = batch_per_step - (50000 % batch_per_step)
        else:
            num_padded = batch_per_step - 50000
        print("eval padded samples:{}".format(num_padded))

        if num_padded != 0:
            white_io = BytesIO()
            Image.new('RGB',(224,224),(255,255,255)).save(white_io, 'JPEG')
            padded_sample = {
                "image": np.array(bytearray(white_io.getvalue()), dtype="uint8"),
                "label": np.array(-1, np.int32)
            }
            sample = [padded_sample for x in range(num_padded)]
            ds_pad = de.PaddedDataset(sample)
            ds_imagefolder = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers)
            ds = ds_pad + ds_imagefolder
            distributeSampler = de.DistributedSampler(num_shards=device_num, shard_id=rank_id, shuffle=False, num_samples=None)
            ds.use_sampler(distributeSampler)
        else:
            ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers, shuffle=False, num_shards=device_num, shard_id=rank_id)

    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(crop_min, 1.0), ratio=(0.75, 1.333)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
            C2.TypeCast(mstype.float16)
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)
    ds = ds.map(input_columns="image", num_parallel_workers=num_workers, operations=trans)
    ds = ds.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds

import os
import glob

import numpy as np
from mxnet import gluon, nd

from data_loading.dali_loader import get_dali_loader
from mlperf_logger import mllog_event, constants

NUM_EVAL_SAMPLES = 42
VAL_SPLITS = {2: {0: [176, 84, 34, 44, 207, 198, 6, 92, 169, 0, 3, 206, 49, 80, 203, 162, 70, 76, 138, 41, 160],
                  1: [125, 185, 5, 66, 128, 189, 171, 111, 86, 52, 157, 78, 161, 12, 56, 61, 65, 24, 187, 87, 112]},
              3: {0: [84, 185, 128, 207, 92, 111, 157, 3, 80, 12, 65, 70, 41, 87],
                  1: [176, 5, 66, 198, 6, 86, 52, 206, 49, 56, 61, 76, 138, 112],
                  2: [125, 34, 44, 189, 171, 169, 0, 78, 161, 203, 162, 24, 187, 160]},
              4: {0: [185, 5, 189, 171, 52, 157, 12, 56, 24, 187], 1: [84, 34, 198, 6, 0, 3, 80, 203, 76, 138],
                  2: [176, 44, 207, 92, 169, 206, 49, 162, 70, 41, 160],
                  3: [125, 66, 128, 111, 86, 78, 161, 61, 65, 87, 112]},
              5: {0: [5, 34, 92, 111, 161, 49, 76, 24, 160], 1: [185, 44, 6, 86, 78, 80, 70, 187, 112],
                  2: [84, 66, 171, 169, 206, 12, 65, 138], 3: [176, 128, 189, 0, 3, 56, 61, 41],
                  4: [125, 207, 198, 52, 157, 203, 162, 87]},
              7: {0: [44, 66, 86, 3, 76, 24], 1: [34, 128, 52, 206, 70, 187, 41],
                  2: [5, 207, 0, 12, 65, 138, 112], 3: [185, 198, 169, 161, 61, 160], 4: [84, 189, 157, 49, 162],
                  5: [176, 171, 111, 80, 203], 6: [125, 6, 92, 78, 56, 87]},
              8: {0: [66, 128, 78, 161, 87, 112], 1: [44, 207, 206, 49, 70, 160], 2: [34, 198, 3, 80, 65],
                  3: [5, 189, 157, 12, 187], 4: [185, 171, 52, 56, 24], 5: [84, 6, 0, 203, 76],
                  6: [176, 92, 169, 162, 41], 7: [125, 111, 86, 61, 138]},
              14: {0: [6, 92, 41, 87], 1: [171, 111, 138, 112], 2: [189, 86, 187, 160], 3: [198, 169, 24],
                   4: [207, 0, 76], 5: [128, 52, 70], 6: [66, 157, 65], 7: [44, 3, 61], 8: [34, 206, 162],
                   9: [5, 78, 203], 10: [185, 161, 56], 11: [84, 49], 12: [176, 80], 13: [125, 12]},
              16: {0: [111, 86], 1: [92, 169], 2: [6, 0, 160], 3: [171, 52, 112],
                   4: [189, 157, 87], 5: [198, 3, 41], 6: [207, 206, 138], 7: [128, 78, 187],
                   8: [66, 161, 24], 9: [44, 49, 76], 10: [34, 80, 70], 11: [5, 12, 65],
                   12: [185, 56], 13: [84, 203], 14: [176, 162], 15: [125, 61]},
              20: {0: [52, 157, 187], 1: [0, 3], 2: [169, 206], 3: [86, 78, 160], 4: [111, 161, 112], 5: [92, 49, 87],
                   6: [6, 80, 41], 7: [171, 12, 138], 8: [189, 56], 9: [198, 203], 10: [207, 162], 11: [128, 61],
                   12: [66, 65], 13: [44, 70], 14: [34, 76], 15: [5, 24], 16: [185], 17: [84], 18: [176], 19: [125]},
              24: {0: [78, 161], 1: [206, 49], 2: [3, 80], 3: [157, 12], 4: [52, 56], 5: [0, 203],
                   6: [169, 162], 7: [86, 61], 8: [111, 65], 9: [92, 70], 10: [6, 76],
                   11: [171, 24], 12: [189, 187], 13: [198, 138], 14: [207, 41], 15: [128, 87], 16: [66, 112],
                   17: [44, 160], 18: [34], 19: [5], 20: [185], 21: [84], 22: [176], 23: [125]},
              32: {0: [61, 65], 1: [162, 70], 2: [203, 76], 3: [56, 24], 4: [12, 187], 5: [80, 138], 6: [49, 41],
                   7: [161, 87], 8: [78, 112], 9: [206, 160], 10: [3], 11: [157], 12: [52], 13: [0], 14: [169],
                   15: [86], 16: [111], 17: [92], 18: [6], 19: [171], 20: [189], 21: [198], 22: [207], 23: [128],
                   24: [66], 25: [44], 26: [34], 27: [5], 28: [185], 29: [84], 30: [176], 31: [125]},
              40: {0: [87, 112], 1: [41, 160], 2: [138], 3: [187], 4: [24], 5: [76], 6: [70], 7: [65],
                   8: [61], 9: [162], 10: [203], 11: [56], 12: [12], 13: [80], 14: [49], 15: [161],
                   16: [78], 17: [206], 18: [3], 19: [157], 20: [52], 21: [0], 22: [169], 23: [86],
                   24: [111], 25: [92], 26: [6], 27: [171], 28: [189], 29: [198], 30: [207], 31: [128],
                   32: [66], 33: [44], 34: [34], 35: [5], 36: [185], 37: [84], 38: [176], 39: [125]}}


def make_val_split_even(x_val, y_val, num_shards, shard_id):
    if num_shards in VAL_SPLITS.keys():
        val_indices = [str(i).zfill(5) for i in VAL_SPLITS[num_shards][shard_id]]
        x_selected = [x for x in x_val if x.split("_")[-2] in val_indices]
        y_selected = [y for y in y_val if y.split("_")[-2] in val_indices]
        print(f"RANK {shard_id}, VAL CASES {val_indices}")
    elif num_shards >= NUM_EVAL_SAMPLES:
        x_selected = [x_val[shard_id % NUM_EVAL_SAMPLES]]
        y_selected = [y_val[shard_id % NUM_EVAL_SAMPLES]]
        print(f"ShardID {shard_id}, caseID {x_selected}")
    else:
        print(f"OPTIMAL SPLIT FOR {num_shards} SHARDS NOT FOUND.")
        x_selected, y_selected = x_val, y_val

    return x_selected, y_selected


def list_files_with_pattern(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def load_data(path, files_pattern):
    data = sorted(glob.glob(os.path.join(path, files_pattern)))
    assert len(data) > 0, f"Found no data at {path}"
    return data


def get_split(data, train_idx, val_idx):
    train = list(np.array(data)[train_idx])
    val = list(np.array(data)[val_idx])
    return train, val


def get_data_split(path: str, num_shards: int):
    with open("evaluation_cases.txt", "r") as f:
        val_cases_list = f.readlines()
    val_cases_list = [case.rstrip("\n") for case in val_cases_list]
    imgs = load_data(path, "*_x.npy")
    lbls = load_data(path, "*_y.npy")
    assert len(imgs) == len(lbls), f"Found {len(imgs)} volumes but {len(lbls)} corresponding masks"
    imgs_train, lbls_train, imgs_val, lbls_val = [], [], [], []
    for (case_img, case_lbl) in zip(imgs, lbls):
        if case_img.split("_")[-2] in val_cases_list:
            imgs_val.append(case_img)
            lbls_val.append(case_lbl)
        else:
            imgs_train.append(case_img)
            lbls_train.append(case_lbl)
    # diff = (num_shards - len(imgs_train) % num_shards) % num_shards
    # extra_imgs = np.random.choice(imgs_train, size=diff, replace=False)
    # extra_lbls = [img.replace("_x", "_y") for img in extra_imgs]

    mllog_event(key='train_samples', value=len(imgs_train), sync=False)
    mllog_event(key='eval_samples', value=len(imgs_val), sync=False)
    # print(f"DATALOADERS: NUM SHARDS {num_shards}, EXTRA IMAGES {len(extra_imgs)}")

    # limit = len(imgs_train) - diff
    # imgs_train, lbls_train = imgs_train[:limit], lbls_train[:limit]
    # imgs_train.extend(extra_imgs)
    # lbls_train.extend(extra_lbls)
    return imgs_train, imgs_val, lbls_train, lbls_val


class SyntheticDataset(gluon.data.Dataset):
    def __init__(self, channels_in=1, channels_out=3, shape=(128, 128, 128), ctx=None, scalar=False):
        x_shape = tuple(shape) + (channels_in,)
        self.x = nd.random.uniform(shape=(32, *x_shape), dtype=np.float32, ctx=ctx)
        if scalar:
            self.y = nd.random.randint(low=0, high=channels_out-1, shape=(32, *shape), dtype=np.int32, ctx=ctx)
            self.y = nd.expand_dims(self.y, -1)
        else:
            y_shape = tuple(shape) + (channels_out,)
            self.y = nd.random.uniform(shape=(32, *y_shape), dtype=np.float32, ctx=ctx)

    def __len__(self):
        return 64

    def __getitem__(self, idx):
        return self.x[idx % 32], self.y[idx % 32]


def get_data_loaders(flags, data_dir, seed, local_rank, global_rank, train_ranks, eval_ranks, spatial_group_size):
    x_train, x_val, y_train, y_val = get_data_split(data_dir, num_shards=len(train_ranks) // spatial_group_size)
    if global_rank in train_ranks:
        shard_id = global_rank // spatial_group_size
        num_shards = len(train_ranks) // spatial_group_size
        train_dataloader = get_dali_loader(flags, x_train, y_train, mode="train", seed=seed, num_shards=num_shards,
                                           device_id=local_rank, shard_id=shard_id, global_rank=global_rank)
    else:
        train_dataloader = None

    if global_rank in eval_ranks:
        shard_id = (global_rank - eval_ranks[0]) // spatial_group_size
        num_shards = len(eval_ranks) // spatial_group_size
        x_val, y_val = make_val_split_even(x_val, y_val, num_shards=num_shards, shard_id=shard_id)
        # print(f"GLOBAL {global_rank} LOCAL {local_rank} SHARD {shard_id} NUM {num_shards} CASES {x_val}")
        val_dataloader = get_dali_loader(flags, x_val, y_val, mode="validation", seed=seed,
                                         num_shards=1, device_id=local_rank)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def get_dummy_loaders(flags, data_dir, seed, local_rank, global_rank, training_ranks, spatial_group_size):
    if spatial_group_size > 1:
        assert flags.batch_size == 1, f"batch_size must be equal to 1, got {flags.batch_size}"
        assert flags.val_batch_size == 1, f"val_batch_size must be equal to 1, got {flags.val_batch_size}"

    train_dataloader = None
    val_dataloader = None
    if global_rank in training_ranks:
        case_id = str(local_rank).zfill(5)
        create_dummy_dataset(data_dir, case_id=case_id)
        x_train = load_data(data_dir, f"*{case_id}_x.npy")
        y_train = load_data(data_dir, f"*{case_id}_y.npy")
        train_dataloader = get_dali_loader(flags, x_train, y_train, mode="train", seed=seed, num_shards=1,
                                           device_id=local_rank, shard_id=0, global_rank=global_rank)

    return train_dataloader, val_dataloader


def create_dummy_dataset(data_dir, case_id):
    os.makedirs(data_dir, exist_ok=True)
    x = np.random.rand(1, 256, 256, 256).astype(np.float32)
    y = np.random.randint(low=0, high=3, size=(1, 256, 256, 256), dtype=np.uint8)
    np.save(os.path.join(data_dir, f"dummy_{case_id}_x.npy"), x)
    np.save(os.path.join(data_dir, f"dummy_{case_id}_y.npy"), y)

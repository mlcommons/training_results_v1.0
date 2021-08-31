import cupy as cp
import math as m
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import DALIGluonIterator, LastBatchPolicy
import horovod.mxnet as hvd


class ExternalInputIterator(object):
    def __init__(self, batch_size, device_id, shard_id, num_shards, filelist, seed, in_gpu=False):
        self.batch_size = batch_size
        self.filelist = filelist
        self.data_set_len = len(self.filelist)
        self.shard_id = shard_id
        self.device_id = device_id
        self.shard_size = m.ceil(self.data_set_len/num_shards) 
        self.files = self.filelist
        self.n = len(self.files)
        self.rng = np.random.default_rng(seed)
        self.data = []
        self.start_from_idx = (self.shard_id * self.n) // num_shards
        self.order = [i for i in range(len(self.filelist))]
        self.count = 0
        for filename in filelist:
            image = np.load(filename)
            if in_gpu:
                with cp.cuda.Device(device_id):
                    image = cp.asarray(image)
            self.data.append(image)
        self.in_gpu = in_gpu

    def __iter__(self):
        self.i = self.start_from_idx
        self.rng.shuffle(self.order)
        return self

    def __next__(self):
        self.i = (self.i + 1) % self.n
        return self.data[self.order[self.i]] 

    def __len__(self):
        return self.data_set_len

    next = __next__


class BasicPipeline(Pipeline):
    def __init__(self, flags, batch_size, input_shape, device_id=0, seed=0):
        super().__init__(batch_size=batch_size, num_threads=flags.num_workers, device_id=device_id, seed=seed,
                         py_start_method="spawn", exec_pipelined=True, prefetch_queue_depth=2, exec_async=True)
        self.flags = flags
        self.internal_seed = seed
        self.input_shape = flags.input_shape
        self.crop_shape = types.Constant(input_shape, dtype=types.INT64)
        self.axis_names = "DHW"
        self.bool = ops.Cast(dtype=types.DALIDataType.BOOL)
        self.reshape = ops.Reshape(device="cpu", layout="CDHW")
        self.transpose = ops.Transpose(device="gpu", perm=[1, 2, 3, 0])

    @staticmethod
    def random_augmentation(probability, augmented, original):
        condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
        neg_condition = condition ^ True
        return condition * augmented + neg_condition * original

    def reshape_fn(self, img, label):
        img = self.reshape(img)
        label = self.reshape(label)
        return img, label

    def random_flips_fn(self, img, label):
        hflip, vflip, dflip = [fn.random.coin_flip(probability=0.33) for _ in range(3)]
        flips = {"horizontal": hflip, "vertical": vflip, "depthwise": dflip, 'bytes_per_sample_hint': 8388608}
        return fn.flip(img, **flips), fn.flip(label, **flips)

    def gaussian_noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=0.1)
        return self.random_augmentation(0.1, img_noised, img)

    def brightness_fn(self, img):
        brightness_scale = self.random_augmentation(0.1, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def gaussian_blur_fn(self, img):
        img_blured = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.25, 1.5)))
        return self.random_augmentation(0.1, img_blured, img)

    @staticmethod
    def slice_fn(img, start_idx, length):
        return fn.slice(img, start_idx, length, axes=[0], out_of_bounds_policy="pad")

    def biased_crop_fn(self, img, label, use_cached_loader):
        roi_start, roi_end = fn.segmentation.random_object_bbox(label,
                                                                format='start_end',
                                                                foreground_prob=self.flags.oversampling,
                                                                classes=[1, 2],
                                                                background=0,
                                                                k_largest=2,
                                                                seed=self.internal_seed,
                                                                device='cpu',
                                                                cache_objects=True)

        anchor = fn.roi_random_crop(label, roi_start=roi_start, roi_end=roi_end, crop_shape=[1, *self.input_shape])
        anchor = fn.slice(anchor, 1, 3, axes=[0])  # drop channel from anchor
        if use_cached_loader:
            label = label.gpu()
            slice_device = 'gpu'
        else:
            slice_device = 'cpu'
        img, label = fn.slice([img, label], anchor, self.crop_shape, axis_names=self.axis_names,
                              out_of_bounds_policy="pad", device=slice_device)

        return img, label

    def move_to_gpu(self, img, label):
        return img.gpu(), label.gpu()


class TrainNumpyPipeline(BasicPipeline):
    def __init__(self, flags, batch_size, image_list, label_list, num_shards=1, device_id=0, shard_id=0, seed=0,
                 image_iterator=None, label_iterator=None):
        super().__init__(flags=flags, batch_size=batch_size, input_shape=flags.input_shape, device_id=device_id)
        self.oversampling = flags.oversampling
        self.flags = flags
        self.image_iterator = image_iterator
        self.label_iterator = label_iterator
        self.input_x = ops.readers.Numpy(files=image_list,
                                         shard_id=shard_id,
                                         num_shards=num_shards,
                                         seed=seed,
                                         pad_last_batch=False,
                                         shuffle_after_epoch=not flags.stick_to_shard,
                                         stick_to_shard=flags.stick_to_shard,
                                         dont_use_mmap=flags.dont_use_mmap,
                                         prefetch_queue_depth=flags.prefetch_queue_depth,
                                         # read_ahead=True,
                                         bytes_per_sample_hint=313997472)
        self.input_y = ops.readers.Numpy(files=label_list,
                                         shard_id=shard_id,
                                         num_shards=num_shards,
                                         seed=seed,
                                         pad_last_batch=False,
                                         shuffle_after_epoch=not flags.stick_to_shard,
                                         stick_to_shard=flags.stick_to_shard,
                                         dont_use_mmap=flags.dont_use_mmap,
                                         prefetch_queue_depth=flags.prefetch_queue_depth,
                                         # read_ahead=True,
                                         bytes_per_sample_hint=78499368)

    def define_graph(self):

        if self.flags.use_cached_loader:
            image = fn.external_source(source=self.image_iterator, no_copy=True, name="ReaderX",
                                       layout='CDHW', batch=False, device='gpu')
            label = fn.external_source(source=self.label_iterator, no_copy=True, name="ReaderY",
                                       layout='CDHW', batch=False, device='cpu')
        else:
            image = self.input_x(name="ReaderX")
            label = self.input_y(name="ReaderY")

        # Volumetric augmentations
        # Check - external_source should already do this
        if not self.flags.use_cached_loader:
            image, label = self.reshape_fn(image, label)

        image, label = self.biased_crop_fn(image, label, self.flags.use_cached_loader)

        if not self.flags.use_cached_loader:
            image, label = self.move_to_gpu(image, label)

        image, label = self.random_flips_fn(image, label)

        # Intensity augmentations
        image = self.brightness_fn(image)
        image = self.gaussian_noise_fn(image)
        image = self.transpose(image)
        label = self.transpose(label)

        return image, label


class ValNumpyPipeline(BasicPipeline):
    def __init__(self, flags, batch_size, image_list, label_list, num_shards=1, device_id=0, seed=0):
        super().__init__(flags=flags, batch_size=batch_size, input_shape=flags.val_input_shape, device_id=device_id)
        self.input_x = ops.readers.Numpy(files=image_list,
                                         shard_id=0,  # device_id,
                                         num_shards=1,  # num_shards,
                                         seed=seed,
                                         pad_last_batch=False,
                                         random_shuffle=False)
        self.input_y = ops.readers.Numpy(files=label_list,
                                         shard_id=0,  # device_id,
                                         num_shards=1,  # num_shards,
                                         seed=seed,
                                         pad_last_batch=False,
                                         random_shuffle=False)

    def define_graph(self):
        image = self.input_x(name="ReaderX")
        label = self.input_y(name="ReaderY")

        # Volumetric augmentations
        image, label = self.reshape_fn(image, label)
        image, label = self.move_to_gpu(image, label)

        image = self.transpose(image)
        label = self.transpose(label)

        return image, label


class DaliGluonIterator(DALIGluonIterator):
    def __init__(self, pipe: Pipeline, num_shards: int, mode: str, reader_name: str):
        lbp = LastBatchPolicy.FILL if mode == "train" else LastBatchPolicy.PARTIAL
        super().__init__(pipelines=[pipe],
                         reader_name=reader_name,
                         auto_reset=True,
                         size=-1,
                         last_batch_padded=False,
                         last_batch_policy=lbp)
        self.mode = mode
        if mode == "train":
            self.dataset_size = 168
        else:
            self.dataset_size = pipe.epoch_size(reader_name)
        self.batch_size = pipe.batch_size
        self.num_shards = num_shards
        self.shard_size = m.ceil(self.dataset_size / num_shards)

    def __len__(self):
        return self.shard_size // self.batch_size

    def __next__(self):
        out = super().__next__()[0]
        return out


def get_buffer(batch_size, input_shape, device_id):
    shape = tuple(input_shape) + (1,)
    return [nd.empty((batch_size, *shape), ctx=mx.gpu(device_id), dtype=np.float32),
            nd.empty((batch_size, *shape), ctx=mx.gpu(device_id), dtype=np.uint8)]


class ScatterInputIterator:
    def __init__(self, iterator, batch_size, device_id, shard_id, flags, global_rank):
        self.iterator = iterator
        self.input_batch_size = batch_size
        self._input_buffer = None
        self._output_buffer = get_buffer(batch_size, flags.input_shape, device_id)
        self._input_offset = 0
        self._output_offset = 0
        self._batch_count = 0
        self._root_count = 0
        self._local_rank = device_id
        self.spatial_group_size = flags.spatial_group_size
        self._spatial_group_id = shard_id // flags.spatial_group_size
        self.shard_id = shard_id
        self._spatial_group_rank = global_rank % flags.spatial_group_size
        splits = np.zeros(hvd.size())
        s = global_rank - (global_rank % flags.spatial_group_size)
        splits[s: s + flags.spatial_group_size] = int(flags.input_shape[0] // flags.spatial_group_size)
        self._splits = splits.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if ((self._input_buffer is None) or (self._input_offset >= self.batch_size)) \
                and self._spatial_group_rank == self._root_count:
            try:
                self._input_buffer = self.iterator.__next__()
            except StopIteration:
                self._input_buffer = self.iterator.__next__()
            self._input_offset = 0
        if self._output_offset >= self.shard_size():
            self._output_offset = 0
            raise StopIteration
        if self._spatial_group_rank == self._root_count:
            for o, i in zip(self._output_buffer, self._input_buffer):
                i.slice_axis(out=o, axis=0,
                             begin=self._input_offset,
                             end=self._input_offset + self.batch_size)
            data = nd.moveaxis(self._output_buffer[0], 0, 1)
            lbl = nd.moveaxis(self._output_buffer[1], 0, 1)
            self._input_offset += self.batch_size

            data_buff = hvd.alltoall(data, splits=list(self._splits), name="data_scatter")
            lbl_buff = hvd.alltoall(lbl, splits=list(self._splits), name="lbl_scatter")
        else:
            # DNHWC
            data_buff = hvd.alltoall(
                nd.empty((128 // self.spatial_group_size, self.batch_size, 128, 128, 1), ctx=mx.gpu(self._local_rank)),
                splits=[0] * hvd.size(), name="data_scatter")
            lbl_buff = hvd.alltoall(
                nd.empty((128 // self.spatial_group_size, self.batch_size, 128, 128, 1), ctx=mx.gpu(self._local_rank),
                         dtype=np.uint8), splits=[0] * hvd.size(), name="lbl_scatter")
        self._output_offset += self.batch_size

        self._root_count = (self._root_count + 1) % self.spatial_group_size
        data = nd.moveaxis(data_buff, 1, 0)
        lbl = nd.moveaxis(lbl_buff, 1, 0)
        return [data, lbl]

    def reset(self):
        self.iterator.reset()

    @property
    def batch_size(self):
        return self.iterator.batch_size

    @property
    def size(self):
        return self.iterator.size

    def epoch_size(self, pipeline_id=0):
        return self.iterator._pipes[pipeline_id].epoch_size

    def shard_size(self):
        return self.iterator.shard_size

    def __len__(self):
        return self.iterator.shard_size // self.output_batch_size


class RateMatchInputIterator:
    def __init__(self, iterator, input_batch_size, output_batch_size, device_id, flags):
        self.iterator = iterator
        self.input_batch_size = input_batch_size
        self.output_batch_size = output_batch_size
        self._input_buffer = None
        self._output_buffer = get_buffer(output_batch_size, flags.input_shape, device_id)
        self._input_offset = 0
        self._output_offset = 0
        self._batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self._input_buffer is None) or (self._input_offset >= self.input_batch_size):
            try:
                self._input_buffer = self.iterator.__next__()
            except StopIteration:
                self._input_buffer = self.iterator.__next__()
            self._input_offset = 0

        if self._output_offset >= self.shard_size():
            self._output_offset = 0
            raise StopIteration

        for o, i in zip(self._output_buffer, self._input_buffer):
            i.slice_axis(out=o, axis=0,
                         begin=self._input_offset,
                         end=self._input_offset + self.output_batch_size)

        self._input_offset += self.output_batch_size
        self._output_offset += self.output_batch_size
        return self._output_buffer

    def reset(self):
        self.iterator.reset()

    @property
    def batch_size(self):
        return self.iterator.batch_size

    @property
    def size(self):
        return self.iterator.iterator.size

    def epoch_size(self, pipeline_id=0):
        return self.iterator._pipes[pipeline_id].epoch_size

    def shard_size(self):
        return self.iterator.shard_size

    def __len__(self):
        return self.iterator.shard_size // self.output_batch_size


def get_dali_loader(flags,
                    image_list,
                    label_list,
                    mode: str = "train",
                    num_shards: int = 1,
                    device_id: int = 0,
                    shard_id: int = 0,
                    global_rank=None,
                    seed=None) -> DALIGluonIterator:
    input_batch_size = flags.batch_size * flags.input_batch_multiplier
    output_batch_size = flags.batch_size
    if seed is None:
        raise ValueError("Seed is not set")
    if mode == "train":

        if flags.use_cached_loader:
            eii_image = ExternalInputIterator(batch_size=1,
                                              device_id=device_id,
                                              shard_id=shard_id,
                                              num_shards=num_shards,
                                              filelist=image_list,
                                              seed=seed,
                                              in_gpu=True)
            eii_label = ExternalInputIterator(batch_size=1,
                                              device_id=device_id,
                                              shard_id=shard_id,
                                              num_shards=num_shards,
                                              filelist=label_list,
                                              seed=seed,
                                              in_gpu=False)
                                              
        else:
            eii_image = None
            eii_label = None

        pipe = TrainNumpyPipeline(flags,
                                  batch_size=flags.batch_size * flags.input_batch_multiplier,
                                  image_list=image_list,
                                  label_list=label_list,
                                  num_shards=num_shards,
                                  device_id=device_id,
                                  shard_id=shard_id,
                                  seed=seed,
                                  image_iterator=eii_image,
                                  label_iterator=eii_label)
        reader_name = ""

    else:
        pipe = ValNumpyPipeline(flags,
                                batch_size=1,
                                image_list=image_list,
                                label_list=label_list,
                                num_shards=num_shards,
                                device_id=device_id,
                                seed=seed)
        reader_name = "ReaderX"

    pipe.build()

    dali_iter = DaliGluonIterator(pipe, num_shards, mode, reader_name=reader_name)
    if mode == "train":
        dali_iter = RateMatchInputIterator(iterator=dali_iter,
                                           input_batch_size=input_batch_size,
                                           output_batch_size=output_batch_size,
                                           device_id=device_id,
                                           flags=flags)

    return dali_iter

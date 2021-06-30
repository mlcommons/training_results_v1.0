import os
from time import time
import numpy as np
from scipy import signal
from tqdm import tqdm

from mxnet import nd
from mpi4py import MPI

NUM_EVAL_SAMPLES = 42


def evaluate(flags, model, loader, sw_inference, score_fn, ctx, eval_comm, epoch=0):
    rank = eval_comm.Get_rank()
    # world_size = eval_comm.Get_size()
    if flags.load_ckpt_path:
        model.load_parameters(os.path.join(flags.load_ckpt_path, "best_model.params"), ctx=ctx)
    scores = []
    t0 = time()
    if rank // flags.spatial_group_size < NUM_EVAL_SAMPLES:
        if sw_inference.cache_dataset and sw_inference.cache:
            if rank == 0:
                print(f"EVALUATION FROM CACHE")
            for i, batch in enumerate(tqdm(sw_inference.cache, disable=(rank != 0) or not flags.verbose)):
                image, label, cache = batch
                output, label = sw_inference.run_from_cache(image=image,
                                                            label=label,
                                                            cache=cache,
                                                            model=model.network,
                                                            overlap=flags.overlap)
                scores.append(score_fn(output, label).asnumpy())
        else:
            for i, batch in enumerate(tqdm(loader, disable=(rank != 0) or not flags.verbose)):
                image, label = batch
                output, label = sw_inference.run(inputs=image,
                                                 label=label,
                                                 model=model.network,
                                                 overlap=flags.overlap,
                                                 padding_mode=flags.pad_mode,
                                                 padding_val=-2.2)
                scores.append(score_fn(output, label).asnumpy())
    else:
        scores = [(0.0, 0.0), (0.0, 0.0)]

    scores = np.array(scores).sum(axis=0).astype(np.float32)
    result = np.zeros_like(scores).astype(np.float32)
    eval_comm.Allreduce([scores, MPI.FLOAT], [result, MPI.FLOAT], MPI.SUM)
    scores = result / (NUM_EVAL_SAMPLES * flags.spatial_group_size)

    eval_metrics = {"epoch": epoch,
                    "L1 dice": scores[-2],
                    "L2 dice": scores[-1],
                    "mean_dice": (scores[-1] + scores[-2]) / 2
                    }
    if rank == 0:
        print(f"EVALUATION TIME: {round(time() - t0, 3)} s.")

    return eval_metrics


def pad_input(volume, roi_shape, strides, padding_mode, padding_val, dim=3):
    """
    mode: constant, reflect, replicate, circular
    """
    image_shape = volume.shape[1:-1]
    bounds = [(strides[i] - image_shape[i] % strides[i]) % strides[i] for i in range(dim)]
    bounds = [bounds[i] if (image_shape[i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i]
              for i in range(dim)]
    paddings = [0, 0,
                0, 0,
                bounds[0] // 2, bounds[0] - bounds[0] // 2,
                bounds[1] // 2, bounds[1] - bounds[1] // 2,
                bounds[2] // 2, bounds[2] - bounds[2] // 2]

    volume = nd.reshape(volume, shape=(volume.shape[0], 1, *image_shape))
    padded_volume = nd.pad(volume, pad_width=paddings, mode=padding_mode, constant_value=padding_val)

    padded_volume = nd.reshape(padded_volume, shape=(volume.shape[0], *padded_volume.shape[2:], 1))
    paddings = [0, 0,
                bounds[0] // 2, bounds[0] - bounds[0] // 2,
                bounds[1] // 2, bounds[1] - bounds[1] // 2,
                bounds[2] // 2, bounds[2] - bounds[2] // 2,
                0, 0]
    return padded_volume, paddings


def gaussian_kernel(n, std, dtype, ctx):
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()
    return nd.array(gaussian3D, dtype=dtype, ctx=ctx)


def get_norm_patch(mode, roi_shape, dtype, ctx):
    if mode == "constant":
        norm_patch = nd.ones(shape=roi_shape, dtype=dtype, ctx=ctx)
    elif mode == "gaussian":
        kernel = gaussian_kernel(roi_shape[0], 0.125*roi_shape[0], dtype=dtype, ctx=ctx)
        norm_patch = nd.stack(kernel, kernel, kernel, axis=-1)
    else:
        raise ValueError("Unknown mode. Available modes are {constant, gaussian}.")
    return norm_patch


class SlidingWindow:
    def __init__(self, batch_size, mode, roi_shape, ctx, precision=np.float16, data_precision=np.float32,
                 cache_dataset=False, local_rank=0, spatial_group_size=1):
        self.batch_size = batch_size
        self.mode = mode
        self.roi_shape = roi_shape
        self.precision = precision
        self.data_precision = data_precision
        self.ctx = ctx
        self.norm_patch = get_norm_patch(mode, roi_shape, precision, ctx)
        self.batch_buffer, self.output_buffer = self.get_buffers()
        self.local_rank = local_rank
        self.spatial_group_size = spatial_group_size
        self.cache_dataset = cache_dataset
        self.cache = []

    def get_buffers(self):
        n = 144
        batch_shape = (n + self.batch_size, *self.roi_shape, 1)
        output_shape = (n + self.batch_size, *self.roi_shape, 3)
        batch = nd.zeros(shape=batch_shape, dtype=self.data_precision, ctx=self.ctx)
        output = nd.zeros(shape=output_shape, dtype=self.precision, ctx=self.ctx)
        return batch, output

    def run(self, inputs, label, model, overlap=0.5, padding_mode="constant", padding_val=0.0):
        image_shape = list(inputs.shape[1:-1])
        dim = len(image_shape)
        strides = [int(self.roi_shape[i] * (1 - overlap)) for i in range(dim)]

        bounds = [image_shape[i] % strides[i] for i in range(dim)]
        bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]

        inputs = inputs[:,
                 bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                 bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                 bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2),
                 :
                 ]
        label = label[:,
                bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2),
                :
                ]

        inputs, paddings = pad_input(inputs, self.roi_shape, strides, padding_mode, padding_val)
        padded_shape = inputs.shape[1:-1]
        size = [(padded_shape[i] - self.roi_shape[i]) // strides[i] + 1 for i in range(dim)]
        shape = (1, *padded_shape, 3)
        result = nd.zeros(shape=shape, dtype=self.precision, ctx=self.ctx)
        norm_map = nd.zeros_like(result)

        # Acquire patches
        count = 0
        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    self.batch_buffer[count] = inputs[0,
                                               i:(self.roi_shape[0] + i),
                                               j:(self.roi_shape[1] + j),
                                               k:(self.roi_shape[2] + k),
                                               :]
                    count += 1

        # Run inference
        # if self.use_spatial_loader:
        #     start = (self.local_rank % self.spatial_group_size) * (self.roi_shape[0] // self.spatial_group_size)
        #     stop = (self.local_rank % self.spatial_group_size + 1) * (self.roi_shape[0] // self.spatial_group_size)
        # else:
        start, stop = 0, self.roi_shape[0]
        batch_pad = self.batch_size - count % self.batch_size if count % self.batch_size else 0

        if self.cache_dataset:
            cache = {"image_shape": image_shape, "padded_shape": padded_shape, "count": count, "paddings": paddings}
            self.cache.append((self.batch_buffer[:count + batch_pad, start:stop].copy(), label.copy(), cache))

        for i in range(0, count + batch_pad, self.batch_size):
            self.output_buffer[i:i+self.batch_size] = \
                model(self.batch_buffer[i:i + self.batch_size, start:stop]) * self.norm_patch

        # Split to correct places
        count = 0
        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    result[
                    0,
                    i:(self.roi_shape[0] + i),
                    j:(self.roi_shape[1] + j),
                    k:(self.roi_shape[2] + k), :] += self.output_buffer[count]
                    norm_map[
                    0,
                    i:(self.roi_shape[0] + i),
                    j:(self.roi_shape[1] + j),
                    k:(self.roi_shape[2] + k), :] += self.norm_patch
                    count += 1

        result /= norm_map

        result = result[
                 :,
                 paddings[2]: image_shape[0] + paddings[2],
                 paddings[4]: image_shape[1] + paddings[4],
                 paddings[6]: image_shape[2] + paddings[6],
                 :]

        return result, label

    def run_from_cache(self, image, label, cache, model, overlap=0.5):
        image_shape = cache["image_shape"]
        padded_shape = cache["padded_shape"]
        count = cache["count"]
        paddings = cache["paddings"]

        dim = len(image_shape)
        strides = [int(self.roi_shape[i] * (1 - overlap)) for i in range(dim)]
        size = [(padded_shape[i] - self.roi_shape[i]) // strides[i] + 1 for i in range(dim)]
        shape = (1, *padded_shape, 3)
        result = nd.zeros(shape=shape, dtype=self.precision, ctx=self.ctx)
        norm_map = nd.zeros_like(result)

        # Run inference
        # if self.use_spatial_loader:
        #     start = (self.local_rank % self.spatial_group_size) * (self.roi_shape[0] // self.spatial_group_size)
        #     stop = (self.local_rank % self.spatial_group_size + 1) * (self.roi_shape[0] // self.spatial_group_size)
        # else:
        # start, stop = 0, self.roi_shape[0]
        batch_pad = self.batch_size - count % self.batch_size if count % self.batch_size else 0
        for i in range(0, count + batch_pad, self.batch_size):
            self.output_buffer[i:i+self.batch_size] = \
                model(image[i:i + self.batch_size]) * self.norm_patch

        # Split to correct places
        count = 0
        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    result[
                    0,
                    i:(self.roi_shape[0] + i),
                    j:(self.roi_shape[1] + j),
                    k:(self.roi_shape[2] + k), :] += self.output_buffer[count]
                    norm_map[
                    0,
                    i:(self.roi_shape[0] + i),
                    j:(self.roi_shape[1] + j),
                    k:(self.roi_shape[2] + k), :] += self.norm_patch
                    count += 1

        result /= norm_map

        result = result[
                 :,
                 paddings[2]: image_shape[0] + paddings[2],
                 paddings[4]: image_shape[1] + paddings[4],
                 paddings[6]: image_shape[2] + paddings[6],
                 :]

        return result, label

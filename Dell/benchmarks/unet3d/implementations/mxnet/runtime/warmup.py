import numpy as np
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet.contrib import amp

from runtime.distributed import sync_training_and_evaluation


def train(flags, model, comm, train_comm, eval_comm,
          transfer_comm, train_ranks, eval_ranks, ctx):
    # spatial_size = 128//flags.spatial_group_size if flags.use_spatial_loader else 128
    shape = (flags.batch_size, 128, 128, 128, 1)
    image = nd.random.uniform(shape=shape, dtype=np.float32, ctx=ctx)
    label = nd.random.randint(low=0, high=3, shape=shape, dtype=np.int32, ctx=ctx).astype(np.uint8)

    for i in range(flags.warmup_iters):
        if flags.static_cast:
            image = image.astype(dtype='float16')

        with autograd.record():
            loss_value = model(image, label)
            if flags.amp:
                with amp.scale_loss(loss_value, model.trainer) as scaled_loss:
                    autograd.backward(scaled_loss)
            elif flags.static_cast:
                scaled_loss = loss_value * flags.static_loss_scale
                autograd.backward(scaled_loss)
            else:
                loss_value.backward()

        model.trainer.step(image.shape[0])
        loss_value.asnumpy()  # to prevent hang
    nd.waitall()

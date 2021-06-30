import numpy as np
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet.contrib import amp

from runtime.distributed import sync_training_and_evaluation


def train(flags, model, train_loader, comm, train_comm, eval_comm,
          transfer_comm, train_ranks, eval_ranks, ctx):

    rank = comm.Get_rank()
    stop_training = False
    # if rank in train_ranks:
    #     model.dummy_trainer.set_learning_rate(0.0)
    spatial_size = 128//flags.spatial_group_size if flags.use_spatial_loader else 128
    shape = (flags.batch_size, spatial_size, 128, 128, 1)
    image = nd.random.uniform(shape=shape, dtype=np.float32, ctx=ctx)
    label = nd.random.randint(low=0, high=3, shape=shape, dtype=np.int32, ctx=ctx).astype(np.uint8)

    for i in range(10):
        if flags.static_cast:
            image, label = image.astype(dtype='float16'), label.astype(dtype='float16')

        with autograd.record():
            loss_value = model(image, label)
            if flags.amp:
                with amp.scale_loss(loss_value, model.trainer) as scaled_loss:
                    autograd.backward(scaled_loss)
            else:
                loss_value.backward()


'''
    for cycle in range(0, 2):
        if rank in train_ranks:
            for training_epoch in range(0, 4):
                for batch in train_loader:
                    image, label = batch
                    if flags.static_cast:
                        image, label = image.astype(dtype='float16'), label.astype(dtype='float16')

                    with autograd.record():
                        loss_value = model(image, label)
                        if flags.amp:
                            with amp.scale_loss(loss_value, model.trainer) as scaled_loss:
                                autograd.backward(scaled_loss)
                        else:
                            loss_value.backward()

                    # model.dummy_trainer.step(image.shape[0] / flags.spatial_group_size)

        # Sync training and eval nodes
        # if flags.nodes_for_eval:
        #     stop_training, model = sync_training_and_evaluation(flags, comm, eval_comm, transfer_comm,
        #                                                         rank, model, train_ranks, eval_ranks,
        #                                                         cycle, stop_training, ctx)
'''

import numpy as np
import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.contrib import amp
from mxnet.lr_scheduler import MultiFactorScheduler
import horovod.mxnet as hvd

from model.losses import DiceCELoss
from model.layers import InputBlock, DownsampleBlock, UpsampleBlock, OutputBlock, SplitBlock, GatherBlock


class SpatialNetwork(gluon.HybridBlock):
    def __init__(self, n_classes, spatial_group_size, local_rank, comm):
        super(SpatialNetwork, self).__init__()
        filters = [32, 64, 128, 256, 320]
        spatial_params = dict(spatial_group_size=spatial_group_size,
                              local_rank=local_rank,
                              comm=comm)
        with self.name_scope():
            self.input_split = SplitBlock(spatial_group_size=spatial_group_size)
            self.input_block = InputBlock(1, filters[0], spatial=True, **spatial_params)

            self.down0 = DownsampleBlock(filters[0], filters[1], index=0, spatial=True, **spatial_params)
            #self.gather_block = GatherBlock(spatial_group_size=spatial_group_size)
            self.down1 = DownsampleBlock(filters[1], filters[2], index=1, spatial=True, **spatial_params)
            self.gather_block = GatherBlock(spatial_group_size=spatial_group_size)
            self.down2 = DownsampleBlock(filters[2], filters[3], index=2, spatial=False, **spatial_params)
            self.down3 = DownsampleBlock(filters[3], filters[4], index=3, spatial=False, **spatial_params)

            self.bottleneck = DownsampleBlock(filters[4], filters[4], index=4, spatial=False, **spatial_params)

            self.up0 = UpsampleBlock(filters[4], filters[4], index=0, spatial=False, **spatial_params)
            self.up1 = UpsampleBlock(filters[4], filters[3], index=1, spatial=False, **spatial_params)
            self.split_block = SplitBlock(spatial_group_size=spatial_group_size)
            self.up2 = UpsampleBlock(filters[3], filters[2], index=2, spatial=True, **spatial_params)
            #self.split_block = SplitBlock(spatial_group_size=spatial_group_size)
            self.up3 = UpsampleBlock(filters[2], filters[1], index=3, spatial=True, **spatial_params)
            self.up4 = UpsampleBlock(filters[1], filters[0], index=4, spatial=True, **spatial_params)

            self.output = OutputBlock(filters[0], n_classes, spatial=True, spatial_group_size=spatial_group_size)
            self.output_gather = GatherBlock(spatial_group_size=spatial_group_size)

    def hybrid_forward(self, F, x):  # H,    C
        x = self.input_split(x)
        skip0 = self.input_block(x)  # 128, 32
        skip1 = self.down0(skip0)  # 64,  64
        skip2 = self.down1(skip1)  # 32, 128
        gather2 = self.gather_block(skip2)
        skip3 = self.down2(gather2)  # 16, 256
        skip4 = self.down3(skip3)  # 8,  320

        x = self.bottleneck(skip4)  # 4,  320

        x = self.up0(x, skip4)  # 8,  320
        x = self.up1(x, skip3)  # 16, 256
        x = self.split_block(x)
        x = self.up2(x, skip2)  # 32, 128
        x = self.up3(x, skip1)  # 64,  64
        x = self.up4(x, skip0)  # 128, 32

        x = self.output(x)  # 128,  4
        x = self.output_gather(x)
        return x


class Network(gluon.HybridBlock):
    def __init__(self, n_classes):
        super(Network, self).__init__()
        filters = [32, 64, 128, 256, 320]
        with self.name_scope():
            self.input_block = InputBlock(1, filters[0])
            self.down0 = DownsampleBlock(filters[0], filters[1], index=0)
            self.down1 = DownsampleBlock(filters[1], filters[2], index=1)
            self.down2 = DownsampleBlock(filters[2], filters[3], index=2)
            self.down3 = DownsampleBlock(filters[3], filters[4], index=3)
            self.bottleneck = DownsampleBlock(filters[4], filters[4], index=4)
            self.up0 = UpsampleBlock(filters[4], filters[4], index=0)
            self.up1 = UpsampleBlock(filters[4], filters[3], index=1)
            self.up2 = UpsampleBlock(filters[3], filters[2], index=2)
            self.up3 = UpsampleBlock(filters[2], filters[1], index=3)
            self.up4 = UpsampleBlock(filters[1], filters[0], index=4)
            self.output = OutputBlock(filters[0], n_classes)

    def hybrid_forward(self, F, x):     # H,    C
        skip0 = self.input_block(x)     # 128, 32
        skip1 = self.down0(skip0)       # 64,  64
        skip2 = self.down1(skip1)       # 32, 128
        skip3 = self.down2(skip2)       # 16, 256
        skip4 = self.down3(skip3)       # 8,  320

        x = self.bottleneck(skip4)      # 4,  320

        x = self.up0(x, skip4)          # 8,  320
        x = self.up1(x, skip3)          # 16, 256
        x = self.up2(x, skip2)          # 32, 128
        x = self.up3(x, skip1)          # 64,  64
        x = self.up4(x, skip0)          # 128, 32

        x = self.output(x)              # 128,  4
        return x


class Unet3D(gluon.HybridBlock):
    def __init__(self, n_classes, spatial_group_size, local_rank, comm):
        super(Unet3D, self).__init__()
        self.channel_axis = -1
        self.loss = DiceCELoss(to_onehot_y=True, use_softmax=True, include_background=False)
        self.trainer = None
        self.dummy_trainer = None
        if spatial_group_size == 1:
            self.network = Network(n_classes)
        else:
            self.network = SpatialNetwork(n_classes, spatial_group_size, local_rank, comm)

    def hybrid_forward(self, F, x, y_true):     # H,    C
        y_pred = self.network(x).astype(np.float32)
        if autograd.is_training():
            return self.loss(y_pred, y_true)
        return y_pred

    def init(self, flags, ctx, world_size, steps_per_epoch, is_training_rank, cold_init=True, warmup_iters = 0):
        # self.collect_params().initialize(init=init.Xavier(magnitude=1.0), ctx=ctx)
        force_reinit = not cold_init
        self.collect_params().initialize(ctx=ctx, force_reinit=force_reinit)
        self.hybridize(static_alloc=True, static_shape=True)
        self.network.hybridize(static_alloc=True, static_shape=flags.spatial_group_size == 1)
        if world_size > 1:
            hvd.broadcast_parameters(self.collect_params(), root_rank=0)

        if is_training_rank and cold_init:
            lr_scheduler = None
            if len(flags.lr_decay_epochs) > 0 or flags.lr_warmup_epochs > 0:
                step = [step * steps_per_epoch + warmup_iters for step in flags.lr_decay_epochs] or [1 + warmup_iters]
                lr_scheduler = MultiFactorScheduler(step=step,
                                                    base_lr=flags.learning_rate,
                                                    factor=flags.lr_decay_factor if len(step) > 0 else 1.0,
                                                    warmup_steps=warmup_iters + flags.lr_warmup_epochs * steps_per_epoch,
                                                    warmup_begin_lr=flags.init_learning_rate)
            optimizer = get_optimizer(flags, lr_scheduler)
            self.trainer = hvd.DistributedTrainer(self.collect_params(), optimizer, num_groups=1,
                                                  gradient_predivide_factor=flags.grad_predivide_factor)
            if flags.amp:
                amp.init_trainer(self.trainer)


def get_optimizer(flags, lr_scheduler=None):
    multi_precision = flags.amp or flags.static_cast
    optim_kwargs = dict(learning_rate=flags.learning_rate,
                        multi_precision=multi_precision,
                        lr_scheduler=lr_scheduler,
                        rescale_grad=1.0/flags.static_loss_scale if flags.static_cast else 1.0)
    if flags.optimizer == "adam":
        optim = mx.optimizer.Adam()
    elif flags.optimizer == "nadam":
        optim = mx.optimizer.Nadam(**optim_kwargs)
    elif flags.optimizer == "sgd":
        optim = mx.optimizer.SGD(**optim_kwargs, momentum=flags.momentum)
    elif flags.optimizer == "nag":
        optim = mx.optimizer.NAG(**optim_kwargs, momentum=flags.momentum)
    elif flags.optimizer == "lamb":
        optim = mx.optimizer.LAMB(**optim_kwargs, beta1=flags.lamb_betas[0], beta2=flags.lamb_betas[1])
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim



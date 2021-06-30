import math
from mxnet.base import check_call, _LIB, c_array, _Null
from mxnet.gluon import nn, HybridBlock
import ctypes
from mpi4py import MPI
import mxnet as mx
import numpy as np

USE_MPI4PY = True

anti_gc = []


def handler_bytes():
    return 64


# def _init_gbn_buffers(bn_group_per_node, num_nodes, c, local_rank, comm=None):
#     assert bn_group_per_node >= 1, 'bn_group can\'t be smaller than 1'
#     if bn_group_per_node == 1:
#         return _Null
#
#     print('bn group per node:', bn_group_per_node)
#
#     global_comm = MPI.COMM_WORLD # TO REPLACE
#     local_comm = global_comm.Split_type(MPI.COMM_TYPE_SHARED)
#     # local_comm =
#     local_gpus = local_comm.Get_size()
#     xbuf_ptr = (ctypes.c_void_p * (local_gpus + 4))()
#     # local_rank = hvd.local_rank()
#     handler = np.zeros(handler_bytes(), dtype=np.byte)
#     check_call(_LIB.MXInitXBufSingle(local_rank, xbuf_ptr, handler.ctypes.data_as(ctypes.c_void_p)))
#
#     handlers = np.asarray([np.zeros(handler_bytes(), dtype=np.byte)]*local_gpus)
#     local_comm.Allgather([handler, handler_bytes(), MPI.BYTE], [handlers, handler_bytes(), MPI.BYTE])
#     check_call(_LIB.MXOpenIpcHandles(local_rank, bn_group_per_node, xbuf_ptr, handlers.ctypes.data_as(ctypes.c_void_p)))
#
#     anti_gc.append(xbuf_ptr)
#
#     # allocate buffers for GBN.
#     check_call(_LIB.MXAllocGBNBuffers(local_rank, num_nodes, bn_group_per_node, c, 2, xbuf_ptr))
#
#     return ctypes.addressof(xbuf_ptr)


def _init_gbn_buffers(bn_group, local_rank, comm):
    assert bn_group >= 1, 'bn_group can\'t be smaller than 1'
    if bn_group == 1:
        return _Null

    sync_depth = int(math.log2(bn_group))  # required sync steps
    # print('sync_depth:', sync_depth)
    # global_comm = MPI.COMM_WORLD
    # local_comm = global_comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_gpus = local_comm.Get_size()
    xbuf_ptr = (ctypes.c_void_p * local_gpus)()
    handler = np.zeros(handler_bytes(), dtype=np.byte)
    check_call(_LIB.MXInitXBufSingle(local_rank, sync_depth, xbuf_ptr, handler.ctypes.data_as(ctypes.c_void_p)))
    #handlers = np.asarray([np.zeros(handler_bytes(), dtype=np.byte)]*local_gpus)
    #local_comm.Allgather([handler, handler_bytes(), MPI.BYTE], [handlers, handler_bytes(), MPI.BYTE])
    #check_call(_LIB.MXOpenIpcHandles(rank, local_gpus, sync_depth, xbuf_ptr, handlers.ctypes.data_as(ctypes.c_void_p)))

    handlers = np.zeros(handler_bytes()*local_gpus, dtype=np.byte)
    local_comm.Allgather([handler, handler_bytes(), MPI.BYTE], [handlers, handler_bytes(), MPI.BYTE])
    check_call(_LIB.MXOpenIpcHandles(local_rank, local_gpus, sync_depth, xbuf_ptr, handlers.ctypes.data_as(ctypes.c_void_p)))

    anti_gc.append(xbuf_ptr)
    return ctypes.addressof(xbuf_ptr)


class GroupInstanceNorm(HybridBlock):
    """
    Batch normalization layer (Ioffe and Szegedy, 2014) with GBN support.
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    Parameters
    ----------
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    bn_group : int, default 1
        Batch norm group size. if bn_group>1 the layer will sync mean and variance between
        all GPUs in the group. Currently only groups of 1, 2 and 4 are supported

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, in_channels=0, axis=-1, scale=True, center=True,
                 spatial_group_size=1, local_rank=0, comm=None, act_type='relu', **kwargs):
        super(GroupInstanceNorm, self).__init__(**kwargs)
        assert spatial_group_size in [1, 2, 4, 8]
        assert comm is not None
        if in_channels != 0:
            self.in_channels = in_channels

        # set parameters.
        self.c_max = 256

        self.xbuf_ptr = _init_gbn_buffers(bn_group=spatial_group_size, local_rank=local_rank, comm=comm)

        self.instance_norm = nn.InstanceNormV2(in_channels=in_channels,
                                               axis=axis,
                                               scale=scale,
                                               center=center,
                                               act_type=act_type,
                                               xbuf_ptr=self.xbuf_ptr,
                                               xbuf_group=spatial_group_size
                                               )

        # BN has the parameters bn_group and xbuf_ptr.  For IN, I called the equivalent params xbuf_group and xbuf_ptr.


    # def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
    #     return F.BatchNorm(data=x, gamma=gamma, beta=beta,
    #                        moving_mean=running_mean, moving_var=running_var,
    #                        bn_group=self.bn_group,
    #                        bn_group_per_node=self.bn_group_per_node,
    #                        rank=self.rank, node=self.node, device=self.device,
    #                        num_nodes=self.num_nodes, node_size=self.node_size,
    #                        xbuf_ptr=self.xbuf_ptr,
    #                        act_type=self.act_type,
    #                        name='fwd', **self._kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.instance_norm(x)


# def SymGroupBatchNorm(x, bn_group, **kwargs):
#     xbuf_ptr = _init_gbn_buffers(bn_group)
#     return mx.sym.BatchNorm(x, bn_group=bn_group, xbuf_ptr=xbuf_ptr, **kwargs)

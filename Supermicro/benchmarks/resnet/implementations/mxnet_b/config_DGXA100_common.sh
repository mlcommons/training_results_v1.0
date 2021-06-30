export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export DALI_DONT_USE_MMAP=0 # 0 for /raid and 1 for lustre
export MXNET_EXTENDED_NORMCONV_SUPPORT=1 # supports Arch 80 NormConv fusion

## System config params
export DGXNGPU=8
export DGXNSOCKET=2
export DGXSOCKETCORES=64
export DGXHT=2  # HT is on is 2, HT off is 1
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=1
export HOROVOD_CYCLE_TIME=0.1
export MXNET_OPTIMIZER_AGGREGATION_SIZE=54
export MXNET_ENABLE_CUDA_GRAPHS=1
# MxNet PP BN Heuristic
export MXNET_CUDNN_NHWC_BN_HEURISTIC_FWD=1
export MXNET_CUDNN_NHWC_BN_HEURISTIC_BWD=1
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD=1
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD=1

# Enable SHARP
#export NCCL_COLLNET_ENABLE=1

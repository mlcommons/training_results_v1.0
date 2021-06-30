export OMPI_MCA_btl="^openib" #To prevent deadlock between Horovd and NCCL at 96 nodes
export DALI_DONT_USE_MMAP=0 # 0 for /raid and 1 for lustre
export MXNET_EXTENDED_NORMCONV_SUPPORT=1 # supports Arch 80 NormConv fusion

## System config params
export PGNGPU=4
export PGNSOCKET=2
export PGSOCKETCORES=64
export PGHT=2  # HT is on is 2, HT off is 1
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=1
export HOROVOD_CYCLE_TIME=0.2
export NCCL_SOCKET_IFNAME="enp97s0f1"
export MXNET_OPTIMIZER_AGGREGATION_SIZE=54

#source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

## DL params
export OPTIMIZER="nag"
export BATCH_SIZE="4"
export VAL_BATCH_SIZE="4"
export LR="4.0"
export LR_WARMUP_EPOCHS="1000"
export MAX_EPOCHS=${MAX_EPOCHS:-4000}
export START_EVAL_AT=1000
export LAYOUT="NDHWC"
export LOADER="dali"
export QUALITY_THRESHOLD="0.908"
export DATADIR="/fs/sjc1-lcl01/ent/joc/datasets/images/msd/preprocessed/kits19_padded"
export INPUT_BATCH_MULTIPLIER=1
export NUM_WORKERS=4
export TARGET_DIR=${TARGET_DIR:-"/tmp"}

export MXNET_USE_TENSORRT=0
export MXNET_EXEC_ENABLE_ADDTO=1
#export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
#export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=25
export MXNET_UPDATE_ON_KVSTORE=0
export MXNET_GPU_WORKER_NTHREADS=1
export MXNET_GPU_COPY_NTHREADS=1
export OMP_NUM_THREADS=1
#export MXNET_HOROVOD_NUM_GROUPS=20

export OMPI_MCA_btl=^openib
#export NCCL_MAX_RINGS=8
#export NCCL_BUFFSIZE=2097152
#export NCCL_NET_GDR_READ=1
#export HOROVOD_CYCLE_TIME=0.1
#export HOROVOD_FUSION_THRESHOLD=67108864
#export HOROVOD_NUM_NCCL_STREAMS=1
export HOROVOD_BATCH_D2D_MEMCOPIES=1
export HOROVOD_GROUPED_ALLREDUCES=1

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
#WALLTIME_MINUTES=240
#export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=16
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
export NCCL_SOCKET_IFNAME=

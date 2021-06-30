#source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

## DL params
export OPTIMIZER="nag"
export BATCH_SIZE="4"
export VAL_BATCH_SIZE="4"
export LR="1"
export LR_WARMUP_EPOCHS="4"
export MAX_EPOCHS=10
export START_EVAL_AT=20
export LAYOUT="NDHWC"

#export HOROVOD_CYCLE_TIME=0.1
#export HOROVOD_FUSION_THRESHOLD=67108864
#export HOROVOD_NUM_NCCL_STREAMS=1
#export MXNET_HOROVOD_NUM_GROUPS=20
#export NHWC_BATCHNORM_LAUNCH_MARGIN=32
#export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
#export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=25
#
#export NCCL_MAX_RINGS=8
#export NCCL_BUFFSIZE=2097152
#export NCCL_NET_GDR_READ=1

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=40
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=1
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
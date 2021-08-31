source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

## DL params -- 13k
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="204"
export KVSTORE="horovod"
export LR="12.8"
export WARMUP_EPOCHS="9"
export EVAL_OFFSET="0" #Targeting epoch number 44, 48 ....
export EVAL_PERIOD="4"
export WD="2.5e-05"
export MOM="0.94"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS=${NUMEPOCHS:-"48"}

export NETWORK="resnet-v1b-stats-fl"
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_BWD=0
export MXNET_CUDNN_NHWC_BN_ADD_HEURISTIC_FWD=0

export DALI_PREFETCH_QUEUE="3"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="12288"
export DALI_PREALLOCATE_WIDTH="5980"
export DALI_PREALLOCATE_HEIGHT="6430"
#DALI buffer presizing hints
export DALI_DECODER_BUFFER_HINT="1315942" #1196311*1.1
export DALI_CROP_BUFFER_HINT="165581" #150528*1.1
export DALI_TMP_BUFFER_HINT="44446041" #871491*batch_size
export DALI_NORMALIZE_BUFFER_HINT="441549" #401408*1.1

# Default is no NCCL and BWD overlap
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=2
export MXNET_HOROVOD_NUM_GROUPS=1
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999

## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:40:00

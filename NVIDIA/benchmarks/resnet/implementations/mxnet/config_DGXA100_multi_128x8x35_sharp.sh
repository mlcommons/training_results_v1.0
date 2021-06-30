source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh
## DL params --35k
export OPTIMIZER="sgdwfastlars"
export BATCHSIZE="35"
export KVSTORE="horovod"
export LR="21.9"
export WARMUP_EPOCHS="17"
export EVAL_OFFSET="3" #Targeting epoch# 59 (total 60)
export EVAL_PERIOD="4"
export WD="0.000025"
export MOM="0.94"
export LARSETA="0.001"
export LABELSMOOTHING="0.1"
export LRSCHED="pow2"
export NUMEPOCHS=${NUMEPOCHS:-"60"}

export NETWORK="resnet-v1b-fl"

export DALI_PREFETCH_QUEUE="3"
export DALI_NVJPEG_MEMPADDING="256"
export DALI_CACHE_SIZE="12288"
export INPUT_BATCH_MULTIPLIER="8"
export DALI_HW_DECODER_LOAD="0.7"
export DALI_PREALLOCATE_WIDTH="5980"
export DALI_PREALLOCATE_HEIGHT="6430"
#DALI buffer presizing hints
export DALI_DECODER_BUFFER_HINT="1315942" #1196311*1.1
export DALI_CROP_BUFFER_HINT="165581" #150528*1.1
export DALI_TMP_BUFFER_HINT="30502185" #871491*batch_size
export DALI_NORMALIZE_BUFFER_HINT="441549" #401408*1.1

# Default is no NCCL and BWD overlap
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_NCCL_STREAMS=1
export MXNET_HOROVOD_NUM_GROUPS=1
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=999

export SBATCH_NETWORK=sharp

## System run parms
export DGXNNODES=128
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:40:00

## System config params
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXNGPU=4
export DGXSOCKETCORES=32
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

export GRAD_ACCUMULATION_STEPS=4

## Run specific params
export DATADIR="/raid/datasets/rnnt/LibriSpeech/"
export BATCHSIZE=256
export EVAL_BATCHSIZE=676
export GRAD_ACCUMULATION_STEPS=2 
export WALLTIME=04:00:00
export MAX_SYMBOL=300
export EPOCH=80
export DATA_CPU_THREADS=16

## Opt flag
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
export BATCH_EVAL_MODE=cg_unroll_pipeline
export APEX_LOSS=fp16
export APEX_JOINT=pack
export AMP_LVL=2
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp16
export DIST_LAMB=false
export MULTILAYER_LSTM=false
export IN_MEM_FILE_LIST=true
export ENABLE_PREFETCH=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export TOKENIZED_TRANSCRIPT=true
export TRAIN_MANIFESTS='/datasets/LibriSpeech/librispeech-train-clean-100-wav-tokenized.json
                        /datasets/LibriSpeech/librispeech-train-clean-360-wav-tokenized.json
                        /datasets/LibriSpeech/librispeech-train-other-500-wav-tokenized.json'
export VAL_MANIFESTS='/datasets/LibriSpeech/librispeech-dev-clean-wav-tokenized.json'
export NCCL_SOCKET_IFNAME=

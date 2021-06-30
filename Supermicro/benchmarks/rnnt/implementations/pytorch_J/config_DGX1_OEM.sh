## System config params
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

## Run specific params
export DATADIR="/raid/datasets/rnnt/LibriSpeech/"
export BATCHSIZE=128
export EVAL_BATCHSIZE=169
export GRAD_ACCUMULATION_STEPS=8
export MAX_SYMBOL=300
export EPOCH=100
export DATA_CPU_THREADS=8

## Opt flag
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
export BATCH_EVAL_MODE=no_cg
export APEX_LOSS=fp16
export APEX_JOINT=pack
export AMP_LVL=2
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp16
export DIST_LAMB=false
export MULTILAYER_LSTM=false
export ENABLE_PREFETCH=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export TOKENIZED_TRANSCRIPT=true
export TRAIN_MANIFESTS='/datasets/LibriSpeech/librispeech-train-clean-100-wav-tokenized.json
                        /datasets/LibriSpeech/librispeech-train-clean-360-wav-tokenized.json
                        /datasets/LibriSpeech/librispeech-train-other-500-wav-tokenized.json'
export VAL_MANIFESTS='/datasets/LibriSpeech/librispeech-dev-clean-wav-tokenized.json'

## Wall time
if [[ "${DRYRUN}" == 1 ]]; then
    export WALLTIME=10:00:00
else
	export WALLTIME=02:00:00
fi

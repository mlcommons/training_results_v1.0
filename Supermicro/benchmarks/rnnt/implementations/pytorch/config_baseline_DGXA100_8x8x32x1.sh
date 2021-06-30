## System config params
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXNGPU=8
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
export GRAD_ACCUMULATION_STEPS=1

## Run specific params
export DATADIR="/raid/datasets/rnnt/LibriSpeech/"
export BATCHSIZE=32
export EVAL_BATCHSIZE=2
export WALLTIME=01:00:00
export VAL_FREQUENCY=1
export MAX_SYMBOL=300
export EPOCH=90
export SEED=$RANDOM
export LR=0.007
export WEIGHTS_INIT_SCALE=0.5
export DATA_CPU_THREADS=8


## DL params
export BATCHSIZE=4
export GRADIENT_STEPS=1
#export INIT_LOSS_SCALE=16384
export LR=0.00288293
export MAX_SAMPLES_TERMINATION=6000000
export MAX_STEPS=110000
export OPT_LAMB_BETA_1=0.88
export OPT_LAMB_BETA_2=0.88
export START_WARMUP_STEP=-76
export WEIGHT_DECAY_RATE=0.0166629
export WARMUP_STEPS=287

export EXTRA_PARAMS="--dense_seq_output --use_cuda_graph"
export PHASE=2

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:60:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1

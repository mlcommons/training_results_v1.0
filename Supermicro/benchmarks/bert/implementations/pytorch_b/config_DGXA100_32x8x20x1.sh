## DL params
export BATCHSIZE=20
export GRADIENT_STEPS=1
export LR=1.7e-3
export MAX_SAMPLES_TERMINATION=6500000
export MAX_STEPS=${MAX_STEPS:-990}
export OPT_LAMB_BETA_1=0.87
export OPT_LAMB_BETA_2=0.97
export WARMUP_UPDATES=0.000

export EXTRA_PARAMS="--dense_seq_output"
export PHASE=2
export USE_DDP=1
## System run parms
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:00:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1

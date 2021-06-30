## DL params
# steps500_lr0.004_wup0.1_b1_0.878_b2_0.974
export BATCHSIZE=32
export GRADIENT_STEPS=1
#export INIT_LOSS_SCALE=16384
export LR=0.00288293
export MAX_SAMPLES_TERMINATION=7000000
export MAX_STEPS=600
export OPT_LAMB_BETA_1=0.88
export OPT_LAMB_BETA_2=0.88
export START_WARMUP_STEP=-76
export WEIGHT_DECAY_RATE=0.0166629
export WARMUP_STEPS=287

export EXTRA_PARAMS=""
export PHASE=2

## System run parms
export DGXNNODES=32
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:25:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1

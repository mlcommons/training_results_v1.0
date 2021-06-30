## DL params
export BATCHSIZE=36
export GRADIENT_STEPS=6
export LR=4e-4
export MAX_STEPS=13889
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export WARMUP_UPDATES=0

export EXTRA_PARAMS="--unpad"
export PHASE=2

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
source ${BASH_SOURCE%/*}/config_DGX1_common.sh
## DL params
export BATCHSIZE=3
export GRADIENT_STEPS=1
#export INIT_LOSS_SCALE=16384
export LR=0.002
export MAX_SAMPLES_TERMINATION=7000000
export MAX_STEPS=1141
export OPT_LAMB_BETA_1=0.66
export OPT_LAMB_BETA_2=0.998
export START_WARMUP_STEP=0
export WEIGHT_DECAY_RATE=0.01
export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

export WARMUP_STEPS=100
export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--use_cuda_graph"
export PHASE=2

## System run parms
export DGXNNODES=128
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:70:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1

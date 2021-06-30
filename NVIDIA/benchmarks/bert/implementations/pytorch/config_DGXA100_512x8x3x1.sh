## DL params
export BATCHSIZE=3
export GRADIENT_STEPS=1
export INIT_LOSS_SCALE=4096.0
export LR=0.0031
export MAX_SAMPLES_TERMINATION=12000000
export MAX_STEPS=500
export OPT_LAMB_BETA_1=0.80
export OPT_LAMB_BETA_2=0.925
export START_WARMUP_STEP=-100
export WEIGHT_DECAY_RATE=0.0166629
export WARMUP_STEPS=300
export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--use_cuda_graph"
export PHASE=2
export EVAL_ITER_START_SAMPLES=275000
export EVAL_ITER_SAMPLES=275000

## System run parms
export DGXNNODES=512
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:50:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1
export USE_DDP=1

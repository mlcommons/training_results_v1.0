## DL params
export BATCHSIZE=27
export GRADIENT_STEPS=1
export LR=4.0e-4
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=${MAX_STEPS:-8103}
export OPT_LAMB_BETA_1=0.9
export OPT_LAMB_BETA_2=0.999
export WARMUP_PROPORTION=0.0
export SBATCH_NETWORK=sharp
export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding"
export PHASE=2

## System run parms
export DGXNNODES=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:00:00

## System config params
source ${BASH_SOURCE%/*}/config_DGXA100_common.sh

export CONTAINER_PRELOAD_LUSTRE=1

## DL params
export BATCH_SIZE=55296
#export DGXNGPU=8
export DGXNGPU=4

export CONFIG="dgx_a100_as-2124.json"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:20:00
export OMPI_MCA_btl="^openib"
#export MOUNTS=/raid:/raid

## NCCL WAR
export NCCL_LAUNCH_MODE=PARALLEL

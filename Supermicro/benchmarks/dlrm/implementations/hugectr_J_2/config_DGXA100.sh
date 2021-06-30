## DL params
export BATCH_SIZE=15296
export DGXNGPU=8

export CONFIG="dgx_a100.json"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
#export WALLTIME=00:20:00
export OMPI_MCA_btl="^openib"
export MOUNTS=/raid:/raid

## NCCL WAR
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_SOCKET_IFNAME=

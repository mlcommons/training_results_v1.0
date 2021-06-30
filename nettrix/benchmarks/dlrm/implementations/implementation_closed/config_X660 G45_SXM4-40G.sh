## DL params
export BATCH_SIZE=55296
export DGXNGPU=8

export CONFIG="dgx_a100.json"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:10:00
export OMPI_MCA_btl="^openib"
export MOUNTS=/raid:/raid

export SBATCH_NETWORK=""
export NCCL_SOCKET_IFNAME=

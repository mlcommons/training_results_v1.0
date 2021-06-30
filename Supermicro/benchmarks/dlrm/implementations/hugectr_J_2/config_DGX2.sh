## DL params
export BATCH_SIZE=55296
export DGXNGPU=16

export CONFIG="dgx2h.json"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=00:20:00
export OMPI_MCA_btl="^openib"
export MOUNTS=/raid/datasets:/raid/datasets

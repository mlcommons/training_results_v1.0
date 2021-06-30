## DL params
export NUMEPOCHS=120
export BATCHSIZE=64
export EXTRA_PARAMS='--lr-decay-epochs 80 100 --lr-warmup-epoch=40 --lr=0.0025 --weight-decay=4e-5 --gradient-predivide-factor=4 --input-jpg-decode=cache'

## System run parms
export DGXNNODES=4
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=15
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

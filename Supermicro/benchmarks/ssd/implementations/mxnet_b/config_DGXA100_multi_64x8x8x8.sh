## DL params
export NUMEPOCHS=160
export BATCHSIZE=8
export EVALBATCHSIZE=5
export EXTRA_PARAMS='--lr-decay-epochs 80 100 --lr-warmup-epoch=40 --lr=0.0025 --weight-decay=4e-5 --bn-group=8 --gradient-predivide-factor=64 --input-jpg-decode=cache --input-batch-multiplier=20'

## System run parms
export DGXNNODES=64
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=15
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

## DL params
export BATCHSIZE=2
export EVALBATCHSIZE=40
export NUMEPOCHS=${NUMEPOCHS:-15}
export EXTRA_PARAMS='--val-epochs 10 15 --lr-decay-epochs 60 75 --lr-warmup-epoch=26 --lr=0.0046875 --weight-decay=4e-5 --bn-group=8 --gradient-predivide-factor=32 --input-batch-multiplier=20'

## System run parms
export DGXNNODES=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=15
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

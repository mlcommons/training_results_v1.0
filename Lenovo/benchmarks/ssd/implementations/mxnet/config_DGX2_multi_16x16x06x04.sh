## DL params
export BATCHSIZE=6
export EVALBATCHSIZE=12
export EXTRA_PARAMS='--lr-warmup-epoch=11 --dali-workers=3 --lr=0.002916 --weight-decay=1.7e-4 --bn-group=4 --gradient-predivide-factor=16'

## System run parms
export DGXNNODES=16
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=15
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

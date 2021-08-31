## DL params
export BATCHSIZE=3
export EVALBATCHSIZE=5
export NUMEPOCHS=${NUMEPOCHS:-90}
export EXTRA_PARAMS='--lr-decay-epochs 68 85 --lr-warmup-epoch=35 --lr=0.0035 --weight-decay=4e-5 --bn-group=8 --gradient-predivide-factor=64 --input-jpg-decode=cache --input-batch-multiplier=20'

## System run parms
export DGXNNODES=128
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=10
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

## Enable SHARP
export SBATCH_NETWORK=sharp

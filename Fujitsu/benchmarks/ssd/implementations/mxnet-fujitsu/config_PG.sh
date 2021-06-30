## DL params
export BATCHSIZE=114
export EXTRA_PARAMS='--lr-warmup-epoch=3 --lr=0.003157 --weight-decay=1.3e-4'


## System run parms
export NEXP=5
export PGNNODES=1
export PGSYSTEM=PG
WALLTIME_MINUTES=20
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export PGNGPU=4
export PGSOCKETCORES=64
export PGNSOCKET=2
export PGHT=2         # HT is on is 2, HT off is 1

## DL params
export BATCHSIZE=7
export EVALBATCHSIZE=40
export EXTRA_PARAMS='--lr-warmup-epoch=18 --dali-workers=3 --bn-group=4 --lr=0.003095 --weight-decay=2e-4 --input-batch-multiplier=10 --gradient-predivide-factor=16 --horovod-num-groups=1'

## System run parms
export DGXNNODES=30
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=10
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export NVIDIA_VISIBLE_DEVICES="$(seq 0 2 $(($DGXNGPU * 2 - 1)) | tr '\n' ',' | sed 's/,$//' )"  # 0,2,4,6,...
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2 	# HT is on is 2, HT off is 1

## DL params
export EXTRA_PARAMS=""
export EXTRA_CONFIG='SOLVER.BASE_LR 0.16 SOLVER.MAX_ITER 40000 SOLVER.WARMUP_FACTOR 0.00016 SOLVER.WARMUP_ITERS 1000 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (10500,14000) SOLVER.IMS_PER_BATCH 128 TEST.IMS_PER_BATCH 16 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 8000 NHWC True'

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=04:00:00

## System config params
export DGXNGPU=16
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

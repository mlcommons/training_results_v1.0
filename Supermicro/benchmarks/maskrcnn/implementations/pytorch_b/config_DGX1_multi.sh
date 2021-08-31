: "${SOLVER_MAX_ITER:=40000}"

## DL params
export EXTRA_PARAMS=""
export EXTRA_CONFIG='SOLVER.BASE_LR 0.24 SOLVER.WARMUP_FACTOR 0.000192 SOLVER.WARMUP_ITERS 1250 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (7000,9333) SOLVER.IMS_PER_BATCH 192 TEST.IMS_PER_BATCH 192 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 1000 NHWC True'
export EXTRA_CONFIG="${EXTRA_CONFIG} SOLVER.MAX_ITER $SOLVER_MAX_ITER"

## System run parms
export DGXNNODES=24
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=08:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=20
export DGXNSOCKET=2
export DGXHT=2 	# HT is on is 2, HT off is 1
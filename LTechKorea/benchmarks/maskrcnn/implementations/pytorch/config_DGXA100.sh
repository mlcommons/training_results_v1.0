## DL params
: "${ENABLE_DALI:=False}"
: "${USE_CUDA_GRAPH:=True}"
: "${CACHE_EVAL_IMAGES:=True}"
: "${EVAL_SEGM_NUMPROCS:=10}"
: "${EVAL_MASK_VIRTUAL_PASTE:=True}"
: "${INCLUDE_RPN_HEAD:=True}"
: "${PRECOMPUTE_RPN_CONSTANT_TENSORS:=True}"
: "${DATALOADER_NUM_WORKERS:=1}"
: "${HYBRID_LOADER:=True}"
: "${SOLVER_MAX_ITER:=40000}"


export EXTRA_PARAMS=""
export EXTRA_CONFIG='SOLVER.BASE_LR 0.12 SOLVER.WARMUP_FACTOR 0.000192 SOLVER.WARMUP_ITERS 625 SOLVER.WARMUP_METHOD mlperf_linear SOLVER.STEPS (12000,16000) SOLVER.IMS_PER_BATCH 96 TEST.IMS_PER_BATCH 96 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 12000 MODEL.RPN.FPN_POST_NMS_TOP_N_PER_IMAGE False NHWC True MODEL.RESNETS.TRANS_FUNC FastBottleneckWithFixedBatchNorm'
export EXTRA_CONFIG="${EXTRA_CONFIG} SOLVER.MAX_ITER ${SOLVER_MAX_ITER} DATALOADER.DALI $ENABLE_DALI DATALOADER.DALI_ON_GPU $ENABLE_DALI DATALOADER.CACHE_EVAL_IMAGES $CACHE_EVAL_IMAGES EVAL_SEGM_NUMPROCS $EVAL_SEGM_NUMPROCS USE_CUDA_GRAPH $USE_CUDA_GRAPH EVAL_MASK_VIRTUAL_PASTE $EVAL_MASK_VIRTUAL_PASTE MODEL.BACKBONE.INCLUDE_RPN_HEAD $INCLUDE_RPN_HEAD DATALOADER.NUM_WORKERS $DATALOADER_NUM_WORKERS PRECOMPUTE_RPN_CONSTANT_TENSORS $PRECOMPUTE_RPN_CONSTANT_TENSORS DATALOADER.HYBRID $HYBRID_LOADER"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=100
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

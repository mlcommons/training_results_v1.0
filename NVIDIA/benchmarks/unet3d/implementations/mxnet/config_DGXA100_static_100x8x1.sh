#source $(dirname ${BASH_SOURCE[0]})/config_DGXA100_common.sh

## DL params
export OPTIMIZER="nag"
export BATCH_SIZE="1"
export VAL_BATCH_SIZE="1"
export LR="1.5"
export LR_WARMUP_EPOCHS="1000"
export MAX_EPOCHS=${MAX_EPOCHS:-7000}
export START_EVAL_AT=700
export EVALUATE_EVERY=14
export QUALITY_THRESHOLD="0.908"
export INPUT_BATCH_MULTIPLIER=4
export NUM_WORKERS=4
export ASYNC_PARAMS=" --nodes_for_eval 20 -sgs 8 -ucl"
export EXTRA_PARAMS=${EXTRA_PARAMS:-""}
export PRECISION=${PRECISION:-"--static_cast -sls 1024 -gpf 256 --loss_scale_inc_cycles 70"}

export SBATCH_NETWORK=sharp
#export MXNET_EXEC_ENABLE_ADDTO=1
#export OMP_NUM_THREADS=1
export HOROVOD_CYCLE_TIME=0.1
#export MXNET_HOROVOD_NUM_GROUPS=20
export OMPI_MCA_btl=^openib
#export NCCL_MAX_RINGS=8
#export NCCL_BUFFSIZE=2097152
#export NCCL_NET_GDR_READ=1
#export HOROVOD_FUSION_THRESHOLD=67108864
#export HOROVOD_NUM_NCCL_STREAMS=1
export HOROVOD_BATCH_D2D_MEMCOPIES=1
export HOROVOD_GROUPED_ALLREDUCES=1

## System run parms
export DGXNNODES=100
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
NEXP=${NEXP:-5} # Default number of times to run the benchmark
WALLTIME_MINUTES=24
export WALLTIME=$((${NEXP} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
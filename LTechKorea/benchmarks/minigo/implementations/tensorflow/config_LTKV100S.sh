#!/bin/sh

## Environment variables for multi node runs
export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_FUSION_THRESHOLD=67108864
export HOROVOD_NUM_STREAMS=2

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $0 | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=07:30:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2 	# HT is on is 2, HT off is 1

## Data mount location
export DATADIR=/opt/data/Dataset/training/minigo/ml_perf
#export LOGDIR=/opt/data/Dataset/training/minigo/ml_perf/log

## supress Tensorflow messages
## 3->FATAL, 2->ERROR, 1->WARNING, 0
export TF_CPP_MIN_LOG_LEVEL=3

## Benchmark knobs for this config.
export NUM_GPUS_TRAIN=4
export NUM_ITERATIONS=70

#multiple procs/gpu
export SP_THREADS=2
export PA_SEARCH=2
export PA_INFERENCE=2
export CONCURRENT_GAMES=32
export PROCS_PER_GPU=4

export CONT=mlperf-nvidia:minigo

echo "DGXSYSTEM: ${DGXSYSTEM}"
echo "DGXNGPU: ${DGXNGPU}"
echo "CONT: ${CONT}"
echo "DATADIR: ${DATADIR}"
#echo "LOGDIR: ${LOGDIR}"

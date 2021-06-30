#!/bin/bash

BENCHMARK=${BENCHMARK:-"object_detection"}
DATADIR=${DATADIR:-"/data/coco-2017"} # there should be ./coco2017 and ./torchvision dirs in here
LOGDIR=${LOGDIR:-"LOG"}
NEXP=${NEXP:-5} # Default number of times to run the benchmark

cd ../pytorch-fujitsu
source config_PG.sh && LOGDIR=$LOGDIR DATADIR=$DATADIR CONT=mlperf-fujitsu:$BENCHMARK NEXP=$NEXP ./run_with_docker.sh

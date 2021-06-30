#!/bin/bash

DATADIR=${DATADIR:-"/data/coco-2017"} # there should be ./coco2017 and ./torchvision dirs in here
LOGDIR=${LOGDIR:-"LOG"}
NEXP=${NEXP:-5} # Default number of times to run the benchmark
PULL=${PULL:-0}

cd ../mxnet-fujitsu
source config_PG.sh
CONT="mlperf-fujitsu:single_stage_detector" LOGDIR=$LOGDIR DATADIR=$DATADIR NEXP=$NEXP ./run_with_docker.sh

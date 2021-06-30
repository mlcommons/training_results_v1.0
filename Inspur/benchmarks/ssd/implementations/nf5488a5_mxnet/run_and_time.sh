#!/bin/bash

cd ../mxnet
source config_NF5488A5.sh
DGXSYSTEM="NF5488A5" CONT=mlperf-inspur:ssd DATADIR=/path/to/preprocessed/data LOGDIR=/path/to/logfile ./run_with_docker.sh

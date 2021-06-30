#!/bin/bash

cd ../mxnet
source config_NF5688M6.sh
DGXSYSTEM="NF5688M6" CONT=mlperf-inspur:ssd DATADIR=/path/to/preprocessed/data LOGDIR=/path/to/logfile ./run_with_docker.sh

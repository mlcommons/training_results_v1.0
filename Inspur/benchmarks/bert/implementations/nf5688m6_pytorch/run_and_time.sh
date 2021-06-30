#!/bin/bash

cd ../pytorch
source config_NF5688M6.sh
DGXSYSTEM="NF5688M6" CONT=mlperf-inspur:bert DATADIR=/path/to/2048_shards_uncompressed/ DATADIR_PHASE2=/path/to/2048_shards_uncompressed/ EVALDIR=/path/to/eval_set_uncompressed/ CHECKPOINTDIR=/path/to/checkpoints CHECKPOINTDIR_PHASE1=/path/to/checkpoints_ph1 ./run_with_docker.sh
#!/bin/bash

BENCHMARK=${BENCHMARK:-"single_stage_detector"}

cd ../mxnet-fujitsu
docker build --pull -t mlperf-fujitsu:$BENCHMARK .

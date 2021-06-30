#!/bin/bash

BENCHMARK=${BENCHMARK:-"object_detection"}

cd ../pytorch-fujitsu
docker build --pull -t mlperf-fujitsu:$BENCHMARK .

#!/bin/bash

source config_*.sh
CONT=nvcr.io/nvdlfwea/mlperfv1/rnnt:20210427.pytorch DATADIR=/home/smci/MLPerf-Training/RNNT/datasets/LibriSpeech/ LOGDIR=/home/smci/MLPerf-Training/RNNT/results/ ./run_with_docker.sh

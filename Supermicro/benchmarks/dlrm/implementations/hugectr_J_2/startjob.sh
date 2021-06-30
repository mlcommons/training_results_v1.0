#!/bin/bash
sudo nvidia-smi -pm 1
source config_DGXA100.sh
CONT=nvcr.io/nvdlfwea/mlperfv1/dlrm:20210517.hugectr DATADIR=/home/smci/MLPerf-Training/DLRM_dataset LOGDIR=/home/smci/MLPerf-Training/DLRM/logs/ ./run_with_docker.sh

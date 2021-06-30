#!/bin/bash
source config_DGXA100.sh
CONT=nvcr.io/nvdlfwea/mlperfv1/maskrcnn:20210426.pytorch DATADIR=/home/smci/MLPerf-Training/MaskR-CNN/data LOGDIR=/home/smci/MLPerf-Training/MaskR-CNN/data/coco2017/log/ ./run_with_docker.sh

#!/bin/bash
sudo nvidia-smi -pm 1
source config_DGXA100_conv-dali_1x8x7.sh
CONT=nvcr.io/nvdlfwea/mlperfv1/unet3d:20210517.mxnet DATADIR=/home/smci/MLPerf-Training/U-Net3D/dataset/unet3d/ LOGDIR=/home/smci/MLPerf-Training/U-Net3D/results/ ./run_with_docker.sh

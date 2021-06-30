#!/usr/bin/env bash

COCO_FOLDER=/raid/datasets/coco/coco-2017/coco2017

if [[ ${DGXSYSTEM} == DGXA100* ]]
then
    BACKBONE_FOLDER=/lustre/fsw/mlperf/mlperft-ssd/akiswani/backbone
else
    BACKBONE_FOLDER=/gpfs/fs1/akiswani/workspace/ssd/ssd-backbone
fi

srun \
    --mpi=pmix \
    --container-image="${1}" \
    --container-workdir=/ssd \
    --container-mounts=$(pwd):/ssd,${COCO_FOLDER}:/data/coco2017,${BACKBONE_FOLDER}:/pretrained/mxnet \
    ./run_and_time.sh

#!/bin/sh
# Copyright (c) 2021, Jay Huh - All Rights Reserved
# Author: Jaeyeong Huh <jay.jyhuh@gmail.com>
# Date  : 2021-09-02 14:02:27 KST

DATASET_DIR="$PWD/coco2017"
ANN_FILE="$DATASET_DIR/annotations/instances_train2017.json"
PKL_OUT_FILE="$DATASET_DIR/pkl_coco/instances_train2017.json.pickled"

mkdir -p $DATASET_DIR/pkl_coco

python pickle_coco_annotations.py \
  --root $DATASET_DIR \
  --ann_file $ANN_FILE \
  --pickle_output_file $PKL_OUT_FILE

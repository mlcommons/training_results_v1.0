#!/bin/bash

# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh


set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
echo "running benchmark"

DATASET_DIR=${DATASET_DIR:-"/data"}

SEED=${1:--1}
OPTIMIZER=${OPTIMIZER:-"nag"}
BATCH_SIZE=${BATCH_SIZE:-4}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-4}
LR=${LR:-"4.0"}
MAX_EPOCHS=${MAX_EPOCHS:-4000}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-1000}
QUALITY_THRESHOLD=${QUALITY_THRESHOLD:-"0.908"}
START_EVAL_AT=${START_EVAL_AT:-1000}
EVALUATE_EVERY=20
TARGET_DIR=${TARGET_DIR:-""}


DISTRIBUTED="mpirun --allow-run-as-root --bind-to none --np ${DGXNGPU}"

# run training
${DISTRIBUTED} python main.py \
    --data_dir "${DATASET_DIR}" \
    --epochs "${MAX_EPOCHS}" \
    --quality_threshold "${QUALITY_THRESHOLD}" \
    --batch_size "${BATCH_SIZE}" \
    --evaluate_every "${EVALUATE_EVERY}" \
    --start_eval_at "${START_EVAL_AT}" \
    --lr_warmup_epochs "${LR_WARMUP_EPOCHS}" \
    --optimizer "${OPTIMIZER}" \
    --learning_rate "${LR}" \
    --layout "${LAYOUT}" \
    --amp \
    --loader "${LOADER}" \
    --val_batch_size "${VAL_BATCH_SIZE}" ${STATIC_SHAPES} \
    --num_workers "${NUM_WORKERS}" \
    --input_batch_multiplier "${INPUT_BATCH_MULTIPLIER}"; ret_code=$?

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="image_segmentation"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

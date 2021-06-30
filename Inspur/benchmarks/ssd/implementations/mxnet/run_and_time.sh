#!/bin/bash

# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
BATCHSIZE=${BATCHSIZE:-32}
EVALBATCHSIZE=${EVALBATCHSIZE:-${BATCHSIZE}}
NUMEPOCHS=${NUMEPOCHS:-120}
LOG_INTERVAL=${LOG_INTERVAL:-100}
DATASET_SIZE=${DATASET_SIZE:-117266}
EVAL_DATASET_SIZE=${EVAL_DATASET_SIZE:-5000}
PRETRAINED_BACKBONE=${PRETRAINED_BACKBONE:-"/pretrained/mxnet/resnet34-333f7ec4.pickle"}
DATASET_DIR=${DATASET_DIR:-"/data/coco2017"}
DATA_LAYOUT=${DATA_LAYOUT:-NHWC}
ASYNC_EVAL=${ASYNC_EVAL:-1}

echo "running benchmark"

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
      cluster=''
    if [[ "${DGXSYSTEM}" == DGX2* ]]; then
      cluster='circe'
    fi
    if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
      cluster='selene'
    fi
    CMD=( './bind.sh' "--cluster=${cluster}" '--ib=single' '--' 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( "python" "-u" "-m" "bind_launch" "--nsockets_per_node=${DGXNSOCKET}" \
        "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
fi

PARAMS=(
      --batch-size              "${BATCHSIZE}"
      --eval-batch-size         "${EVALBATCHSIZE}"
      --epochs                  "${NUMEPOCHS}"
      --log-interval            "${LOG_INTERVAL}"
      --coco-root               "${DATASET_DIR}"
      --dataset-size            "${DATASET_SIZE}"
      --eval-dataset-size       "${EVAL_DATASET_SIZE}"
      --pretrained-backbone     "${PRETRAINED_BACKBONE}"
      --data-layout             "${DATA_LAYOUT}"
)

if [[ ${ASYNC_EVAL} -eq 1 ]]; then
    PARAMS+=(
    --async-val
    )
fi

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # TODO(ahmadki): track the apiLog.sh bug and remove the workaround
  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

# run training
${LOGGER:-} "${CMD[@]}" ssd_main_async.py "${PARAMS[@]}" ${EXTRA_PARAMS} ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

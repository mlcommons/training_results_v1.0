#!/bin/bash

# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${LOGDIR:=$(pwd)/results}"

# The following variables variables need to be set
# Base container to be used
readonly docker_image=${CONT:-"nvcr.io/SET_THIS_TO_CORRECT_CONTAINER_TAG"}
# Location of dataset for phase 1
#readonly DATADIR="/SET_THIS_TO_LOCAL_DATASET_DIR/mlperf/dataset/hdf5/2048_shards"
# Location of dataset for phase 2
#readonly DATADIR_PHASE2="/SET_THIS_TO_LOCAL_DATASET_DIR/mlperf/dataset/hdf5/2048_shards"
# Path to where trained checkpoints will be saved on the system
#readonly CHECKPOINTDIR="/SET_THIS_TO_LOCAL_DATASET_DIR/mlperf/checkpoints"
# Path to pretrained Phase1 checkpoint
#readonly CHECKPOINTDIR_PHASE1="/SET_THIS_TO_LOCAL_DATASET_DIR/mlperf/dataset/download"
#readonly EVALDIR="/SET_THIS_TO_LOCAL_DATASET_DIR/mlperf/dataset/hdf5/2048_shards"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=language_model
_cont_mounts=("--volume=${DATADIR}:/workspace/data" "--volume=${DATADIR_PHASE2}:/workspace/data_phase2" "--volume=${CHECKPOINTDIR}:/results" "--volume=${CHECKPOINTDIR_PHASE1}:/workspace/phase1" "--volume=${EVALDIR}:/workspace/evaldata")

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

PHASE1="\
    --train_batch_size=${BATCHSIZE:-10} \
    --learning_rate=${LR:-6e-3} \
    --warmup_proportion=${WARMUP_PROPORTION:-0.0} \
    --max_steps=7038 \
    --num_steps_per_checkpoint=2500 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --input_dir=/workspace/data \
    "
PHASE2="\
    --train_batch_size=${BATCHSIZE:-10} \
    --learning_rate=${LR:-4e-3} \
    --opt_lamb_beta_1=${OPT_LAMB_BETA_1:-0.9} \
    --opt_lamb_beta_2=${OPT_LAMB_BETA_2:-0.999} \
    --warmup_proportion=${WARMUP_PROPORTION:-0.0} \
    --warmup_steps=${WARMUP_STEPS:-0.0} \
    --start_warmup_step=${START_WARMUP_STEP:-0.0} \
    --max_steps=$MAX_STEPS \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --input_dir=/workspace/data_phase2 \
    --init_checkpoint=/workspace/phase1/model.ckpt-28252.pt \
    "
PHASES=( "$PHASE1" "$PHASE2" )

PHASE=${PHASE:-2}

cluster=''

MAX_SAMPLES_TERMINATION=${MAX_SAMPLES_TERMINATION:-14000000}
EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-100000}
EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-100000}

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' '--cpu=exclusive' '--ib=single' '--cluster=${cluster}' '--' 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( 'python' '-u' '-m' 'bind_pyt' "--nsockets_per_node=${DGXNSOCKET}" \
    "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
fi

GRADIENT_STEPS=${GRADIENT_STEPS:-2}
USE_DDP=${USE_DDP:-0}

# Run fixed number of training samples
BERT_CMD="\
    ${CMD[@]} \
    /workspace/bert/run_pretraining.py \
    $PHASE2 \
    --do_train \
    --skip_checkpoint \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=${TARGET_MLM_ACCURACY:-0.720} \
    --weight_decay_rate=${WEIGHT_DECAY_RATE:-0.01} \
    --max_samples_termination=${MAX_SAMPLES_TERMINATION} \
    --eval_iter_start_samples=${EVAL_ITER_START_SAMPLES} --eval_iter_samples=${EVAL_ITER_SAMPLES} \
    --eval_batch_size=16 --eval_dir=/workspace/evaldata \
    --cache_eval_data \
    --output_dir=/results \
    --fp16 --fused_gelu_bias --fused_mha ${EXTRA_PARAMS} \
    --distributed_lamb   --dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-blocks=1  \
    --gradient_accumulation_steps=${GRADIENT_STEPS} \
    --log_freq=0 \
    --bert_config_path=/workspace/phase1/bert_config.json"

if [[ $USE_DDP != 1 || $GRADIENT_STEPS != 1 ]]; then
    BERT_CMD="${BERT_CMD} --allreduce_post_accumulation --allreduce_post_accumulation_fp16"
fi

# Setup container
nvidia-docker run --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
#make sure container has time to finish initialization
sleep 30
docker exec -it "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        docker exec -it "${_cont_name}" python -c "
import mlperf_logger 
from mlperf_logging.mllog import constants 
mlperf_logger.mlperf_submission_log(\"language_model\")"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${_cont_name}" python -c "
from mlperf_logging.mllog import constants 
from mlperf_logger import log_event 
log_event(key=constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
        docker exec -it "${_config_env[@]}" "${_cont_name}" sh -c "./run_and_time.sh \"${BERT_CMD}\" ${SEED:-$RANDOM}"
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done

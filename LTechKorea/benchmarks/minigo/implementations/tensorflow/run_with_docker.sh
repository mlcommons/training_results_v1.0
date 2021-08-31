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
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/raid/datasets/minigo_data_19x19}"
: "${LOGDIR:=$(pwd)/results}"

# Get num-mpi processes to launch from DGXSYSTEM config file
# Our configs run DGXNGPU*PROCS_PER_GPU processes per system-node, and an additional process to communicate between train & selfplay.
SLURM_NTASKS_PER_NODE=$(( 1 + PROCS_PER_GPU * DGXNGPU ))
export SLURM_NTASKS_PER_NODE

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=minigo
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${LOGDIR}:/results")

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(SLURM_NTASKS_PER_NODE)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

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
import mlperf_log_utils
from mlperf_logging.mllog import constants
mlperf_log_utils.mlperf_submission_log(constants.MINIGO)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${_cont_name}" python -c "
from mlperf_logging.mllog import constants
from mlperf_log_utils import log_event
log_event(key=constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
        docker exec -it "${_config_env[@]}" "${_cont_name}" bash -c "mpirun --allow-run-as-root -np $SLURM_NTASKS_PER_NODE ./run_and_time.sh"
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done

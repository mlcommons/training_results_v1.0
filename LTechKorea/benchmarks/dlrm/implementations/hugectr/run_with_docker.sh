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

#SBATCH --job-name dlrm.hugectr
#SBATCH -t 00:30:00

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${MOUNTS:='/raid/datasets:/raid/datasets,/gpfs/fs1:/gpfs/fs1'}"
: "${LOGDIR:=./results}"
# default DLRM_BIND to null because we don't know what user's system actually is
: "${DLRM_BIND:=}"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=dlrm_hugectr
_cont_mounts=("--volume=${DATADIR}:/raid/datasets/criteo/mlperf/40m.limit_preshuffled/" "--volume=${LOGDIR}:${LOGDIR}")

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(DATADIR)
_config_env+=(DATASET_TYPE)
_config_env+=(DGXSYSTEM)
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
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
#make sure container has time to finish initialization
sleep 30
docker exec -it "${_cont_name}" true


# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
  (
    if [[ $CLEAR_CACHES == 1 ]]; then
      bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
      docker exec -it "${_cont_name}" python3 -c "
from mlperf_logging.mllog import constants
from mlperf_logger.utils import log_event
log_event(key=constants.CACHE_CLEAR, value=True)"
    fi
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
    docker exec -it ${_config_env[@]} ${_cont_name} bash ./run_and_time.sh
  ) |& tee "${_logfile_base}_${_experiment_index}.log"
done

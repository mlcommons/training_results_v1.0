#!/bin/bash

# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

# default value for DLRM_BIND only if it is not already defined
: ${DLRM_BIND:="numactl --membind=1,3,5,7"}

START_TIMESTAMP=$(date +%s)

echo "DLRM_BIND is set to \"${DLRM_BIND}\""
${DLRM_BIND} huge_ctr --train ${CONFIG} | tee /tmp/dlrm_hugectr.log

python3 -m mlperf_logger.format_ctr_output --log_path  /tmp/dlrm_hugectr.log \
            --config_file ${CONFIG} --start_timestamp $START_TIMESTAMP

ret_code=${PIPESTATUS[0]}
set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi


result=`grep -i "hit target" /tmp/dlrm_hugectr.log | awk -F " " '{print $(NF-1)}'`
# exit -1 is accuracy is not hit (ignore for API_LOGGING)
if [ -z $result ] && [ ${API_LOGGING:-0} -eq 0 ];
then
  echo "Didn't hit target AUC $AUC_THRESHOLD"
  exit -1
fi


echo "RESULT,DLRM,$result,$USER"

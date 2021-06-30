#!/bin/bash
# runs benchmark and reports time to convergence

START_TIMESTAMP=$(date +%s)
numactl --membind=0,2 huge_ctr --train ${CONFIG} | tee /tmp/dlrm_hugectr.log
#numactl --interleave=all huge_ctr --train ${CONFIG} | tee /tmp/dlrm_hugectr.log

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

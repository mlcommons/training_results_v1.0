#!/bin/bash

TMP_FNAME="/tmp/summarize_tmp.txt"
./mll-status ${1}/* | awk '{ print $5 }' > ${TMP_FNAME}
cat ${TMP_FNAME}
num_runs=`cat ${TMP_FNAME} | wc -l`
num_success=`grep -c "^[0-9]" ${TMP_FNAME}`
num_failure=`grep -c "^[a-z]" ${TMP_FNAME}`
success_rate=`echo "print($num_success/$num_runs)" | python3`
rm ${TMP_FNAME}
echo "=========================================="
echo "number of runs = ${num_runs}"
echo "number of successful runs = ${num_success}"
echo "number of failed runs = ${num_failure}"
echo "success rate = ${success_rate}"

